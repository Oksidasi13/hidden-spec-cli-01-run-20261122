// mend — one command-line tool that applies AI-written edits to a tree of
// text files, as SPEC.md (version 1) specifies. It reads a patch in one of
// four formats (unified diff, SEARCH/REPLACE blocks, V4A, JSON tool calls),
// finds the lines each edit names at the first of three matching tiers,
// and writes every change or none, printing what it changed or why it
// failed. Node 22, standard library alone; run as `node main.ts [options]
// [PATCH]`. In a trial the grader runs `node /bot/main.ts` with
// MAKEFUN_ENV_URL set, and this file hands over to runner.ts (the last
// lines below): keep that hand-over and runner.ts as they are.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

const FORMATS = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

/** An exit-2 answer: the single line it prints. */
class ParseFault {
  readonly line: number;
  constructor(line: number) {
    this.line = line;
  }
}

/** One edit that failed: its code and the path the failure concerns. */
class EditFault {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const isBlank = (s: string): boolean => s.trim() === '';

/** A patch's lines (SPEC.md section 3): split at \n, one trailing \r off
 * each, no empty piece after a final \n. */
function splitPatch(text: string): string[] {
  if (text === '') return [];
  const parts = text.split('\n');
  if (text.endsWith('\n')) parts.pop();
  for (let i = 0; i < parts.length; i++) {
    if (parts[i]!.endsWith('\r')) parts[i] = parts[i]!.slice(0, -1);
  }
  return parts;
}

/** A path is safe when it is not empty, does not start with /, holds no \,
 * and none of its /-separated segments is empty, . or .. (section 3). */
function isSafePath(p: string): boolean {
  if (p === '' || p.startsWith('/') || p.includes('\\')) return false;
  return p.split('/').every((s) => s !== '' && s !== '.' && s !== '..');
}

/** Byte order of paths (UTF-8), for the output rows. */
function byteCmp(a: string, b: string): number {
  return Buffer.compare(Buffer.from(a, 'utf8'), Buffer.from(b, 'utf8'));
}

/** A file as lines: its line break, its lines, whether it ends with one. */
interface FileModel {
  br: '\n' | '\r\n';
  lines: string[];
  finalBreak: boolean;
}

function splitFile(text: string): FileModel {
  if (text === '') return { br: '\n', lines: [], finalBreak: false };
  const firstLf = text.indexOf('\n');
  const br: '\n' | '\r\n' = firstLf > 0 && text[firstLf - 1] === '\r' ? '\r\n' : '\n';
  const finalBreak = text.endsWith(br);
  const lines = text.split(br);
  if (finalBreak) lines.pop();
  return { br, lines, finalBreak };
}

function joinFile(m: FileModel): string {
  if (m.lines.length === 0) return '';
  return m.lines.join(m.br) + (m.finalBreak ? m.br : '');
}

/** The tree as the edits leave it: `pending` holds what changed (null:
 * deleted), the rest is read from disk on demand and never changes. */
class WorkTree {
  private readonly disk = new Map<string, string | null>();
  readonly pending = new Map<string, string | null>();
  readonly moves: Array<{ from: string; to: string }> = [];

  /** The text on disk before the patch (null: no such file). */
  original(p: string): string | null {
    if (!this.disk.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.disk.set(p, v);
    }
    return this.disk.get(p)!;
  }

  /** The text as earlier edits left it (null: gone). */
  read(p: string): string | null {
    return this.pending.has(p) ? this.pending.get(p)! : this.original(p);
  }

  set(p: string, v: string | null): void {
    this.pending.set(p, v);
  }

  /** Put the tree back as it was before one edit ran. */
  restore(pending: Map<string, string | null>, moves: number): void {
    this.pending.clear();
    for (const [k, v] of pending) this.pending.set(k, v);
    this.moves.length = moves;
  }
}

type EditFn = (tree: WorkTree, strict: boolean) => void;

/** The leading spaces and tabs of a line. */
function leadingWs(s: string): string {
  const m = /^[ \t]*/.exec(s);
  return m === null ? '' : m[0];
}

/** A line as a tier compares it: tier 0 whole, tier 1 without trailing
 * spaces and tabs, tier 2 without leading or trailing ones (section 5). */
function tiered(s: string, tier: number): string {
  if (tier === 0) return s;
  if (tier === 1) return s.replace(/[ \t]+$/, '');
  return s.replace(/^[ \t]+/, '').replace(/[ \t]+$/, '');
}

/** Every 0-based position where `needle` occurs in `hay` at `tier`. */
function occurrences(needle: string[], hay: string[], tier: number): number[] {
  const out: number[] = [];
  const n = needle.length;
  if (n === 0 || n > hay.length) return out;
  const norm = needle.map((l) => tiered(l, tier));
  const hayNorm = hay.map((l) => tiered(l, tier));
  for (let i = 0; i + n <= hay.length; i++) {
    let ok = true;
    for (let j = 0; j < n; j++) {
      if (hayNorm[i + j] !== norm[j]) {
        ok = false;
        break;
      }
    }
    if (ok) out.push(i);
  }
  return out;
}

/** The first tier (of `tiers`, in order) at which `needle` occurs in `hay`,
 * with every occurrence there; null when it occurs at none. */
function firstTier(needle: string[], hay: string[], tiers: number[]): { tier: number; at: number[] } | null {
  for (const t of tiers) {
    const at = occurrences(needle, hay, t);
    if (at.length > 0) return { tier: t, at };
  }
  return null;
}

/** One line of N: `ctx` is the index into O of the context line it copies
 * from the file, or -1 for a line the patch adds. */
interface NewLine {
  text: string;
  ctx: number;
}

/** The lines that replace the matched ones: a context line is the file's own
 * matched line, never the patch's; at tier 2 every other line is re-indented
 * from O's first line's leading whitespace to the file's (section 5). */
function buildNew(nl: NewLine[], o: string[], fileLines: string[], pos: number, tier: number): string[] {
  const out: string[] = [];
  if (tier === 2 && o.length > 0) {
    const fLead = leadingWs(fileLines[pos] ?? '');
    const pLead = leadingWs(o[0]!);
    for (const l of nl) {
      if (l.ctx >= 0) {
        out.push(fileLines[pos + l.ctx]!);
        continue;
      }
      const t = l.text;
      if (t !== '' && t.trim() !== '' && t.startsWith(pLead)) out.push(fLead + t.slice(pLead.length));
      else out.push(t);
    }
    return out;
  }
  for (const l of nl) out.push(l.ctx >= 0 ? fileLines[pos + l.ctx]! : l.text);
  return out;
}

/** The tiers an edit tries: tier 0 only with --strict, else all three. */
function tiersFor(strict: boolean): number[] {
  return strict ? [0] : [0, 1, 2];
}

/** Which format a patch is, with --format=auto (section 4). */
function detectFormat(lines: string[]): string | null {
  const first = lines.findIndex((l) => !isBlank(l));
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const c = lines[first]!.trimStart()[0];
  if (c === '[' || c === '{') return 'json';
  if (lines.some((l) => l.trimEnd() === '<<<<<<< SEARCH')) return 'search-replace';
  if (lines.some((l, i) => l.startsWith('diff --git ') || (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')))) return 'unified';
  return null;
}

// ---------------------------------------------------------------------------
// JSON edits (section 9)
// ---------------------------------------------------------------------------

/** One JSON edit object, or null when the value is not one. */
function jsonEditFn(v: unknown): EditFn | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  if (typeof o.path !== 'string') return null;
  const p = o.path;
  if (o.command === 'create') {
    if (typeof o.file_text !== 'string') return null;
    const text = o.file_text;
    return (tree) => {
      if (!isSafePath(p)) throw new EditFault('unsafe', p);
      if (tree.read(p) !== null) throw new EditFault('exists', p);
      tree.set(p, text);
    };
  }
  if (o.command === 'str_replace') {
    if (typeof o.old_str !== 'string' || o.old_str === '') return null;
    if (o.new_str !== undefined && typeof o.new_str !== 'string') return null;
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      if (!isSafePath(p)) throw new EditFault('unsafe', p);
      const before = tree.read(p);
      if (before === null) throw new EditFault('missing', p);
      const nl = before.indexOf('\n');
      const crlf = nl > 0 && before[nl - 1] === '\r';
      const toLf = (s: string): string => (crlf ? s.replaceAll('\r\n', '\n') : s);
      const body = toLf(before);
      const old = toLf(oldStr);
      let count = 0;
      for (let k = body.indexOf(old); k >= 0; k = body.indexOf(old, k + 1)) count += 1;
      if (count === 0) throw new EditFault('no-match', p);
      if (count > 1) throw new EditFault('ambiguous', p);
      const at = body.indexOf(old);
      const out = body.slice(0, at) + toLf(newStr) + body.slice(at + old.length);
      tree.set(p, crlf ? out.replaceAll('\n', '\r\n') : out);
    };
  }
  if (o.command === 'insert') {
    if (typeof o.insert_line !== 'number' || !Number.isInteger(o.insert_line)) return null;
    if (typeof o.new_str !== 'string') return null;
    const lineNo = o.insert_line;
    const add = splitPatch(o.new_str);
    return (tree) => {
      if (!isSafePath(p)) throw new EditFault('unsafe', p);
      const before = tree.read(p);
      if (before === null) throw new EditFault('missing', p);
      const model = splitFile(before);
      if (lineNo < 0 || lineNo > model.lines.length) throw new EditFault('no-match', p);
      const lines = [...model.lines];
      lines.splice(lineNo, 0, ...add);
      tree.set(p, joinFile({ br: model.br, lines, finalBreak: model.finalBreak }));
    };
  }
  return null;
}

/** The JSON edits of a patch: one array, or one edit object a line. */
function readJson(text: string, lines: string[]): EditFn[] {
  const first = lines.findIndex((l) => !isBlank(l));
  if (first < 0) throw new ParseFault(1);
  const edits: EditFn[] = [];
  if (lines[first]!.trimStart().startsWith('[')) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      throw new ParseFault(first + 1);
    }
    if (!Array.isArray(value)) throw new ParseFault(first + 1);
    for (const item of value) {
      const e = jsonEditFn(item);
      if (e === null) throw new ParseFault(first + 1);
      edits.push(e);
    }
    return edits;
  }
  lines.forEach((l, idx) => {
    if (isBlank(l)) return;
    let e: EditFn | null = null;
    try {
      e = jsonEditFn(JSON.parse(l));
    } catch {
      e = null;
    }
    if (e === null) throw new ParseFault(idx + 1);
    edits.push(e);
  });
  return edits;
}

// ---------------------------------------------------------------------------
// Unified diffs (section 6)
// ---------------------------------------------------------------------------

interface Hunk {
  /** The hunk's old start line, 1-based. */
  a: number;
  /** The context and - lines, in body order. */
  o: string[];
  /** The context and + lines, in body order. */
  n: NewLine[];
  /** A \ line follows the hunk's last context or + line. */
  noEol: boolean;
  /** A \ line follows the hunk's last + line. */
  plusNoEol: boolean;
}

interface UniSection {
  gitOld: string | null;
  gitNew: string | null;
  pairOld: string | null;
  pairNew: string | null;
  renameOld: string | null;
  renameNew: string | null;
  hasPair: boolean;
  isCreate: boolean;
  isDelete: boolean;
  hunks: Hunk[];
}

function newUniSection(gitOld: string | null, gitNew: string | null): UniSection {
  return {
    gitOld,
    gitNew,
    pairOld: null,
    pairNew: null,
    renameOld: null,
    renameNew: null,
    hasPair: false,
    isCreate: false,
    isDelete: false,
    hunks: [],
  };
}

/** The path a --- or +++ line names: the text after the four characters, up
 * to the first tab, with surrounding spaces removed and one leading a/ or
 * b/ taken off. */
function diffPath(raw: string): string {
  const tab = raw.indexOf('\t');
  let s = (tab >= 0 ? raw.slice(0, tab) : raw).trim();
  if (s.startsWith('a/') || s.startsWith('b/')) s = s.slice(2);
  return s;
}

/** A hunk body's lines, up to the first line that is not one; empty lines at
 * the end of the body are not part of it. */
function hunkBody(lines: string[], start: number): { body: string[]; next: number } {
  const body: string[] = [];
  let j = start;
  while (j < lines.length) {
    const l = lines[j]!;
    if (l.startsWith('diff --git ')) break;
    if (l.startsWith('--- ') && (lines[j + 1] ?? '').startsWith('+++ ')) break;
    if (l === '' || l.startsWith(' ') || l.startsWith('-') || l.startsWith('+') || l.startsWith('\\')) {
      body.push(l);
      j += 1;
    } else break;
  }
  while (body.length > 0 && body[body.length - 1] === '') body.pop();
  return { body, next: j };
}

/** Fill a hunk's O and N from its body lines. */
function fillHunk(body: string[], hunk: Hunk): void {
  let lastCtxPlus = -1;
  let lastPlus = -1;
  for (let k = 0; k < body.length; k++) {
    const b = body[k]!;
    if (b.startsWith(' ') || b.startsWith('+')) lastCtxPlus = k;
    if (b.startsWith('+')) lastPlus = k;
  }
  hunk.noEol = lastCtxPlus >= 0 && lastCtxPlus + 1 < body.length && body[lastCtxPlus + 1]!.startsWith('\\');
  hunk.plusNoEol = lastPlus >= 0 && lastPlus + 1 < body.length && body[lastPlus + 1]!.startsWith('\\');
  for (const b of body) {
    if (b.startsWith(' ')) {
      const t = b.slice(1);
      hunk.o.push(t);
      hunk.n.push({ text: t, ctx: hunk.o.length - 1 });
    } else if (b.startsWith('-')) {
      hunk.o.push(b.slice(1));
    } else if (b.startsWith('+')) {
      hunk.n.push({ text: b.slice(1), ctx: -1 });
    } else if (b === '') {
      hunk.o.push('');
      hunk.n.push({ text: '', ctx: hunk.o.length - 1 });
    }
    // \ lines only say the file ends without a line break; they add no line.
  }
}

const HUNK_HEAD = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;

/** The rest of one section: its header, then its hunks. Returns the index
 * of the next section's first line, or the patch's end. */
function parseUniSection(lines: string[], i: number, sec: UniSection): number {
  const n = lines.length;
  while (i < n) {
    const line = lines[i]!;
    if (line.startsWith('@@')) {
      if (!HUNK_HEAD.test(line)) throw new ParseFault(i + 1);
      break;
    }
    if (line.startsWith('diff --git ')) return i;
    if (line.startsWith('--- ')) {
      if ((lines[i + 1] ?? '').startsWith('+++ ')) {
        sec.pairOld = diffPath(line.slice(4));
        sec.pairNew = diffPath(lines[i + 1]!.slice(4));
        sec.hasPair = true;
        i += 2;
        continue;
      }
      throw new ParseFault(i + 2);
    }
    if (line.startsWith('+++ ')) {
      i += 1;
      continue;
    }
    if (line.startsWith('rename from ')) {
      if (!sec.hasPair) sec.renameOld = line.slice('rename from '.length).trim();
      i += 1;
      continue;
    }
    if (line.startsWith('rename to ')) {
      if (!sec.hasPair) sec.renameNew = line.slice('rename to '.length).trim();
      i += 1;
      continue;
    }
    if (line.startsWith('new file mode')) {
      sec.isCreate = true;
      i += 1;
      continue;
    }
    if (line.startsWith('deleted file mode')) {
      sec.isDelete = true;
      i += 1;
      continue;
    }
    i += 1;
  }
  while (i < n) {
    const line = lines[i]!;
    if (line.startsWith('diff --git ')) return i;
    if (line.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) return i;
    if (line.startsWith('@@')) {
      const m = HUNK_HEAD.exec(line);
      if (m === null) throw new ParseFault(i + 1);
      const hunk: Hunk = { a: parseInt(m[1]!, 10), o: [], n: [], noEol: false, plusNoEol: false };
      const { body, next } = hunkBody(lines, i + 1);
      fillHunk(body, hunk);
      sec.hunks.push(hunk);
      i = next;
      continue;
    }
    i += 1;
  }
  return i;
}

/** A section's hunks applied in order to a file's text. */
function applyHunks(text: string, hunks: Hunk[], strict: boolean, path: string): string {
  const model = splitFile(text);
  let lines = [...model.lines];
  let finalBreak = model.finalBreak;
  let d = 0;
  const tiers = tiersFor(strict);
  for (const h of hunks) {
    if (h.o.length === 0) {
      const before = lines.length;
      const at = Math.min(h.a + d, before);
      const ins = h.n.map((l) => l.text);
      lines = [...lines.slice(0, at), ...ins, ...lines.slice(at)];
      if (at === before) finalBreak = !h.noEol;
      d += h.n.length;
      continue;
    }
    const found = firstTier(h.o, lines, tiers);
    if (found === null) throw new EditFault('no-match', path);
    const target = h.a - 1 + d;
    let pos = found.at[0]!;
    for (const p of found.at) {
      if (Math.abs(p - target) < Math.abs(pos - target)) pos = p;
    }
    const runsToEnd = pos + h.o.length === lines.length;
    const repl = buildNew(h.n, h.o, lines, pos, found.tier);
    lines = [...lines.slice(0, pos), ...repl, ...lines.slice(pos + h.o.length)];
    if (runsToEnd) finalBreak = !h.noEol;
    d += h.n.length - h.o.length;
  }
  return joinFile({ br: model.br, lines, finalBreak });
}

/** One unified section as an edit. `/dev/null` names no file: it is the
 * creation marker on the old side and the deletion marker on the new one,
 * never a path to check for safety. */
function unifiedSectionEdit(sec: UniSection): EditFn {
  const rawOld = sec.pairOld ?? sec.renameOld ?? sec.gitOld;
  const rawNew = sec.pairNew ?? sec.renameNew ?? sec.gitNew;
  const oldPath = rawOld === '/dev/null' ? null : rawOld;
  const newPath = rawNew === '/dev/null' ? null : rawNew;
  return (tree, strict) => {
    if (oldPath !== null && !isSafePath(oldPath)) throw new EditFault('unsafe', oldPath);
    if (newPath !== null && !isSafePath(newPath)) throw new EditFault('unsafe', newPath);
    if (sec.isCreate || oldPath === null) {
      const target = newPath!;
      if (tree.read(target) !== null) throw new EditFault('exists', target);
      const plus: string[] = [];
      for (const h of sec.hunks) {
        for (const l of h.n) if (l.ctx < 0) plus.push(l.text);
      }
      let noEol = false;
      for (let k = sec.hunks.length - 1; k >= 0; k--) {
        if (sec.hunks[k]!.n.some((l) => l.ctx < 0)) {
          noEol = sec.hunks[k]!.plusNoEol;
          break;
        }
      }
      tree.set(target, joinFile({ br: '\n', lines: plus, finalBreak: !noEol }));
      return;
    }
    if (sec.isDelete || newPath === null) {
      const target = oldPath!;
      if (tree.read(target) === null) throw new EditFault('missing', target);
      tree.set(target, null);
      return;
    }
    const src = oldPath!;
    const text = tree.read(src);
    if (text === null) throw new EditFault('missing', src);
    if (oldPath !== newPath) {
      const dst = newPath!;
      if (tree.read(dst) !== null) throw new EditFault('exists', dst);
      tree.set(src, null);
      tree.set(dst, applyHunks(text, sec.hunks, strict, src));
      tree.moves.push({ from: src, to: dst });
      return;
    }
    tree.set(src, applyHunks(text, sec.hunks, strict, src));
  };
}

/** The unified edits of a patch. */
function readUnified(lines: string[]): EditFn[] {
  const sections: UniSection[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    if (line.startsWith('diff --git ')) {
      const m = /^diff --git a\/(\S+) b\/(\S+)$/.exec(line);
      if (m === null) throw new ParseFault(i + 1);
      const sec = newUniSection(m[1]!, m[2]!);
      sections.push(sec);
      i = parseUniSection(lines, i + 1, sec);
    } else if (line.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) {
      const sec = newUniSection(null, null);
      sec.pairOld = diffPath(line.slice(4));
      sec.pairNew = diffPath(lines[i + 1]!.slice(4));
      sec.hasPair = true;
      sections.push(sec);
      i = parseUniSection(lines, i + 2, sec);
    } else if (line.startsWith('@@')) {
      throw new ParseFault(i + 1);
    } else {
      i += 1;
    }
  }
  if (sections.length === 0) throw new ParseFault(1);
  return sections.map(unifiedSectionEdit);
}

// ---------------------------------------------------------------------------
// SEARCH/REPLACE blocks (section 7)
// ---------------------------------------------------------------------------

interface SrBlock {
  path: string;
  oldLines: string[];
  newLines: string[];
}

/** The SEARCH/REPLACE edits of a patch. */
function readSearchReplace(lines: string[]): EditFn[] {
  const blocks: SrBlock[] = [];
  let prevEnd = 0;
  let prevPath: string | null = null;
  let i = 0;
  while (i < lines.length) {
    if (lines[i]!.trimEnd() !== '<<<<<<< SEARCH') {
      i += 1;
      continue;
    }
    const searchLine = i + 1;
    let path: string | null = null;
    for (let j = i - 1; j >= prevEnd; j--) {
      const t = lines[j]!;
      if (isBlank(t)) continue;
      const trimmed = t.trim();
      if (trimmed.startsWith('```')) continue;
      path = trimmed;
      break;
    }
    if (path === null) {
      if (prevPath === null) throw new ParseFault(searchLine);
      path = prevPath;
    }
    let sep = -1;
    let end = -1;
    for (let k = i + 1; k < lines.length; k++) {
      if (lines[k]!.trimEnd() === '=======') {
        sep = k;
        break;
      }
      if (lines[k]!.trimEnd() === '>>>>>>> REPLACE') break;
    }
    if (sep < 0) throw new ParseFault(searchLine);
    for (let k = sep + 1; k < lines.length; k++) {
      if (lines[k]!.trimEnd() === '>>>>>>> REPLACE') {
        end = k;
        break;
      }
    }
    if (end < 0) throw new ParseFault(searchLine);
    blocks.push({ path, oldLines: lines.slice(i + 1, sep), newLines: lines.slice(sep + 1, end) });
    prevPath = path;
    prevEnd = end + 1;
    i = end + 1;
  }
  if (blocks.length === 0) throw new ParseFault(1);
  return blocks.map((b) => (tree, strict) => {
    if (!isSafePath(b.path)) throw new EditFault('unsafe', b.path);
    if (b.oldLines.length === 0) {
      if (tree.read(b.path) !== null) throw new EditFault('exists', b.path);
      tree.set(b.path, joinFile({ br: '\n', lines: b.newLines, finalBreak: true }));
      return;
    }
    const text = tree.read(b.path);
    if (text === null) throw new EditFault('missing', b.path);
    const model = splitFile(text);
    const found = firstTier(b.oldLines, model.lines, tiersFor(strict));
    if (found === null) throw new EditFault('no-match', b.path);
    if (found.at.length > 1) throw new EditFault('ambiguous', b.path);
    const pos = found.at[0]!;
    const repl = buildNew(b.newLines.map((t) => ({ text: t, ctx: -1 })), b.oldLines, model.lines, pos, found.tier);
    const out = [...model.lines.slice(0, pos), ...repl, ...model.lines.slice(pos + b.oldLines.length)];
    tree.set(b.path, joinFile({ br: model.br, lines: out, finalBreak: model.finalBreak }));
  });
}

// ---------------------------------------------------------------------------
// V4A patches (section 8)
// ---------------------------------------------------------------------------

interface V4aChunk {
  anchor: string | null;
  o: string[];
  n: NewLine[];
  eof: boolean;
}

function isV4aSectionStart(l: string): boolean {
  return l.startsWith('*** Add File: ') || l.startsWith('*** Delete File: ') || l.startsWith('*** Update File: ');
}

/** The V4A edits of a patch. */
function readV4a(lines: string[]): EditFn[] {
  const first = lines.findIndex((l) => !isBlank(l));
  if (first < 0) throw new ParseFault(1);
  if (lines[first]!.trim() !== '*** Begin Patch') throw new ParseFault(first + 1);
  let last = lines.length - 1;
  while (last >= 0 && isBlank(lines[last]!)) last -= 1;
  if (last < 0 || lines[last]!.trim() !== '*** End Patch') throw new ParseFault(lines.length + 1);
  const edits: EditFn[] = [];
  let i = first + 1;
  while (true) {
    while (i < last && isBlank(lines[i]!)) i += 1;
    if (i >= last) break;
    const line = lines[i]!;
    if (line.startsWith('*** Add File: ')) {
      const path = line.slice('*** Add File: '.length).trim();
      i += 1;
      const fileLines: string[] = [];
      while (i < last && !isV4aSectionStart(lines[i]!)) {
        const l = lines[i]!;
        if (l === '') {
          fileLines.push('');
          i += 1;
          continue;
        }
        if (l.startsWith('+')) {
          fileLines.push(l.slice(1));
          i += 1;
          continue;
        }
        throw new ParseFault(i + 1);
      }
      while (fileLines.length > 0 && fileLines[fileLines.length - 1] === '') fileLines.pop();
      edits.push((tree) => {
        if (!isSafePath(path)) throw new EditFault('unsafe', path);
        if (tree.read(path) !== null) throw new EditFault('exists', path);
        tree.set(path, joinFile({ br: '\n', lines: fileLines, finalBreak: true }));
      });
      continue;
    }
    if (line.startsWith('*** Delete File: ')) {
      const path = line.slice('*** Delete File: '.length).trim();
      i += 1;
      while (i < last && isBlank(lines[i]!)) i += 1;
      if (i < last && !isV4aSectionStart(lines[i]!)) throw new ParseFault(i + 1);
      edits.push((tree) => {
        if (!isSafePath(path)) throw new EditFault('unsafe', path);
        if (tree.read(path) === null) throw new EditFault('missing', path);
        tree.set(path, null);
      });
      continue;
    }
    if (line.startsWith('*** Update File: ')) {
      const updateLine = i + 1;
      const path = line.slice('*** Update File: '.length).trim();
      i += 1;
      let moveTo: string | null = null;
      if (i < last && lines[i]!.startsWith('*** Move to: ')) {
        moveTo = lines[i]!.slice('*** Move to: '.length).trim();
        i += 1;
      }
      const chunks: V4aChunk[] = [];
      let cur: V4aChunk | null = null;
      const pushChunk = (): void => {
        if (cur !== null) {
          chunks.push(cur);
          cur = null;
        }
      };
      while (i < last && !isV4aSectionStart(lines[i]!)) {
        const l = lines[i]!;
        if (l.startsWith('@@')) {
          const anchor = l.slice(2).trim();
          pushChunk();
          cur = { anchor: anchor === '' ? null : anchor, o: [], n: [], eof: false };
          i += 1;
          continue;
        }
        if (l === '*** End of File') {
          if (cur === null) cur = { anchor: null, o: [], n: [], eof: false };
          cur.eof = true;
          pushChunk();
          i += 1;
          continue;
        }
        if (l.startsWith('*** Move to: ')) throw new ParseFault(i + 1);
        if (l.startsWith(' ') || l.startsWith('-') || l.startsWith('+') || l === '') {
          if (cur === null) cur = { anchor: null, o: [], n: [], eof: false };
          if (l.startsWith(' ')) {
            const t = l.slice(1);
            cur.o.push(t);
            cur.n.push({ text: t, ctx: cur.o.length - 1 });
          } else if (l.startsWith('-')) {
            cur.o.push(l.slice(1));
          } else if (l.startsWith('+')) {
            cur.n.push({ text: l.slice(1), ctx: -1 });
          } else {
            cur.o.push('');
            cur.n.push({ text: '', ctx: cur.o.length - 1 });
          }
          i += 1;
          continue;
        }
        throw new ParseFault(i + 1);
      }
      pushChunk();
      if (chunks.length > 0) {
        const tail = chunks[chunks.length - 1]!;
        while (
          tail.o.length > 0 &&
          tail.o[tail.o.length - 1] === '' &&
          tail.n.length > 0 &&
          tail.n[tail.n.length - 1]!.ctx === tail.o.length - 1 &&
          tail.n[tail.n.length - 1]!.text === ''
        ) {
          tail.o.pop();
          tail.n.pop();
        }
      }
      if (chunks.length === 0 && moveTo === null) throw new ParseFault(updateLine);
      edits.push(v4aUpdateEdit(path, moveTo, chunks));
      continue;
    }
    throw new ParseFault(i + 1);
  }
  if (edits.length === 0) throw new ParseFault(1);
  return edits;
}

/** One V4A Update File section as an edit. */
function v4aUpdateEdit(path: string, moveTo: string | null, chunks: V4aChunk[]): EditFn {
  return (tree, strict) => {
    if (!isSafePath(path)) throw new EditFault('unsafe', path);
    if (moveTo !== null && !isSafePath(moveTo)) throw new EditFault('unsafe', moveTo);
    const text = tree.read(path);
    if (text === null) throw new EditFault('missing', path);
    if (moveTo !== null && tree.read(moveTo) !== null) throw new EditFault('exists', moveTo);
    const model = splitFile(text);
    let lines = [...model.lines];
    let cursor = 0;
    const tiers = tiersFor(strict);
    for (const chunk of chunks) {
      if (chunk.anchor !== null) {
        let foundAt = -1;
        for (const t of tiers) {
          for (let j = cursor; j < lines.length; j++) {
            if (tiered(lines[j]!, t) === tiered(chunk.anchor, t)) {
              foundAt = j;
              break;
            }
          }
          if (foundAt >= 0) break;
        }
        if (foundAt < 0) throw new EditFault('no-match', path);
        cursor = foundAt + 1;
      }
      if (chunk.o.length === 0) {
        const at = chunk.eof ? lines.length : cursor;
        const ins = chunk.n.map((l) => l.text);
        lines = [...lines.slice(0, at), ...ins, ...lines.slice(at)];
        cursor = at + ins.length;
        continue;
      }
      let found: { tier: number; at: number[] } | null = null;
      for (const t of tiers) {
        const at: number[] = [];
        for (let j = cursor; j + chunk.o.length <= lines.length; j++) {
          if (chunk.eof && j + chunk.o.length !== lines.length) continue;
          let ok = true;
          for (let k = 0; k < chunk.o.length; k++) {
            if (tiered(lines[j + k]!, t) !== tiered(chunk.o[k]!, t)) {
              ok = false;
              break;
            }
          }
          if (ok) at.push(j);
        }
        if (at.length > 0) {
          found = { tier: t, at };
          break;
        }
      }
      if (found === null) throw new EditFault('no-match', path);
      const pos = found.at[0]!;
      const repl = buildNew(chunk.n, chunk.o, lines, pos, found.tier);
      lines = [...lines.slice(0, pos), ...repl, ...lines.slice(pos + chunk.o.length)];
      cursor = pos + repl.length;
    }
    const out = joinFile({ br: model.br, lines, finalBreak: model.finalBreak });
    if (moveTo !== null) {
      tree.set(path, null);
      tree.set(moveTo, out);
      tree.moves.push({ from: path, to: moveTo });
      return;
    }
    tree.set(path, out);
  };
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

function readPatch(kind: string, text: string, lines: string[]): EditFn[] {
  if (kind === 'unified') return readUnified(lines);
  if (kind === 'search-replace') return readSearchReplace(lines);
  if (kind === 'v4a') return readV4a(lines);
  return readJson(text, lines);
}

function run(argv: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let checkOnly = false;
  let strict = false;
  const positional: string[] = [];
  let ended = false;
  for (const a of argv) {
    if (!ended && a === '--') {
      ended = true;
    } else if (!ended && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=')) {
        const v = a.slice('--format='.length);
        if (format !== null || !FORMATS.includes(v)) return { code: 2, out: ['error: usage'] };
        format = v;
      } else if (a === '--check') {
        if (checkOnly) return { code: 2, out: ['error: usage'] };
        checkOnly = true;
      } else if (a === '--strict') {
        if (strict) return { code: 2, out: ['error: usage'] };
        strict = true;
      } else {
        return { code: 2, out: ['error: usage'] };
      }
    } else {
      positional.push(a);
    }
  }
  if (positional.length > 1) return { code: 2, out: ['error: usage'] };
  const patchArg = positional.length === 0 || positional[0] === '-' ? null : positional[0]!;
  let text = '';
  try {
    text = readFileSync(patchArg ?? 0, 'utf8');
  } catch {
    if (patchArg !== null) return { code: 2, out: ['error: no-patch'] };
  }
  const lines = splitPatch(text);
  const kind = format === null || format === 'auto' ? detectFormat(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };
  let edits: EditFn[];
  try {
    edits = readPatch(kind, text, lines);
    if (edits.length === 0) throw new ParseFault(1);
  } catch (e) {
    if (e instanceof ParseFault) return { code: 2, out: [`error: parse ${e.line}`] };
    throw e;
  }
  const tree = new WorkTree();
  const failures: EditFault[] = [];
  for (const edit of edits) {
    const savedPending = new Map(tree.pending);
    const savedMoves = tree.moves.length;
    try {
      edit(tree, strict);
    } catch (e) {
      if (!(e instanceof EditFault)) throw e;
      tree.restore(savedPending, savedMoves);
      failures.push(e);
    }
  }
  if (failures.length > 0) {
    return {
      code: 1,
      out: [...failures.map((f) => `error: ${f.code} ${f.path}`), `failed: ${failures.length} of ${edits.length} edits`],
    };
  }
  interface Row {
    key: string;
    text: string;
  }
  const rows: Row[] = [];
  for (const p of [...tree.pending.keys()].sort(byteCmp)) {
    const was = tree.original(p);
    const now = tree.pending.get(p)!;
    if (was === now) continue;
    if (was === null) rows.push({ key: p, text: `A ${p}` });
    else if (now === null) rows.push({ key: p, text: `D ${p}` });
    else rows.push({ key: p, text: `M ${p}` });
  }
  for (const mv of tree.moves) {
    const di = rows.findIndex((r) => r.text === `D ${mv.from}`);
    const ai = rows.findIndex((r) => r.text === `A ${mv.to}`);
    if (di < 0 || ai < 0) continue;
    rows.splice(Math.max(di, ai), 1);
    rows.splice(Math.min(di, ai), 1);
    rows.push({ key: mv.to, text: `R ${mv.from} -> ${mv.to}` });
  }
  rows.sort((a, b) => byteCmp(a.key, b.key));
  if (!checkOnly) {
    for (const p of [...tree.pending.keys()].sort(byteCmp)) {
      const now = tree.pending.get(p)!;
      if (now === null) rmSync(p, { force: true });
      else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  const texts = rows.map((r) => r.text);
  return { code: 0, out: [...texts, `ok: ${texts.length} ${texts.length === 1 ? 'file' : 'files'}`] };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = run(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}
