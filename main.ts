// A tiny stack virtual machine for make.fun's tiny stack ISA, built to the
// published spec rule for rule: wrapping 32-bit values, a 256-value stack, a
// 64-deep call stack, 256 words of memory, ten traps with their own exit
// codes, and the step and output limits. Each run answers every instance.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// Per-opcode facts indexed by opcode byte, from rows of
// [opcode, immediate length, pops, pushes]; a length of -1 names no opcode.
const IMM = new Int8Array(256).fill(-1);
const POP = new Int8Array(256);
const PSH = new Int8Array(256);
const META = [
  0x00, 0, 0, 0, 0x01, 1, 0, 1, 0x02, 4, 0, 1, 0x03, 0, 1, 0, // HALT, PUSH8, PUSH32, POP
  0x04, 0, 1, 2, 0x05, 0, 2, 2, 0x06, 0, 2, 3, 0x07, 0, 3, 3, // DUP, SWAP, OVER, ROT
  0x17, 0, 1, 1, 0x1b, 0, 1, 1, 0x30, 2, 0, 0, 0x31, 2, 1, 0, // NEG, NOT, JMP, JZ
  0x32, 2, 1, 0, 0x33, 2, 0, 0, 0x34, 0, 0, 0, 0x40, 0, 1, 1, // JNZ, CALL, RET, LOAD
  0x41, 0, 2, 0, 0x50, 0, 1, 0, 0x51, 0, 1, 0, 0x52, 0, 0, 1, // STORE, PRINT, PUTC, IN
  0x53, 0, 1, 0, // EXIT
];
for (let i = 0; i < META.length; i += 4) {
  IMM[META[i]] = META[i + 1];
  POP[META[i]] = META[i + 2];
  PSH[META[i]] = META[i + 3];
}
for (let op = 0x10; op <= 0x22; op++) if (op !== 0x17 && op !== 0x1b && op !== 0x1f) { IMM[op] = 0; POP[op] = 2; PSH[op] = 1; } // arithmetic, comparisons

function run(prog, inp) {
  const n = prog.length;
  const S = new Int32Array(256); // data stack
  const C = new Int32Array(64); // call stack
  const M = new Int32Array(256); // memory, zeroed
  const out = new Uint8Array(65_536);
  let sp = 0, csp = 0, olen = 0, ip = 0, pc = 0, steps = 0;
  const end = (code) => [out.subarray(0, olen), code];
  for (;;) {
    if (steps === 1_000_000) return end(10); // step limit
    if (pc < 0 || pc >= n) return end(9); // pc outside
    const op = prog[pc];
    const imm = IMM[op];
    if (imm < 0) return end(5); // unknown opcode
    if (pc + imm >= n) return end(9); // immediate cut off
    steps++;
    if (sp < POP[op]) return end(2); // stack underflow
    if (sp - POP[op] + PSH[op] > 256) return end(3); // stack overflow
    const at = pc + 1;
    const next = at + imm;
    pc = next;
    switch (op) {
      case 0x00: return end(0); // HALT
      case 0x01: S[sp++] = prog.readInt8(at); break; // PUSH8, sign-extended
      case 0x02: S[sp++] = prog.readInt32LE(at); break; // PUSH32, little-endian
      case 0x03: sp--; break; // POP
      case 0x04: S[sp] = S[sp - 1]; sp++; break; // DUP
      case 0x05: { const t = S[sp - 1]; S[sp - 1] = S[sp - 2]; S[sp - 2] = t; break; } // SWAP
      case 0x06: S[sp] = S[sp - 2]; sp++; break; // OVER
      case 0x07: { const a = S[sp - 3]; S[sp - 3] = S[sp - 2]; S[sp - 2] = S[sp - 1]; S[sp - 1] = a; break; } // ROT: ( a b c -- b c a )
      case 0x17: S[sp - 1] = -S[sp - 1]; break; // NEG (the Int32Array wraps -(-2^31))
      case 0x1b: S[sp - 1] = ~S[sp - 1]; break; // NOT
      case 0x30: case 0x31: case 0x32: { // JMP, JZ, JNZ
        const rel = prog.readInt16LE(at);
        if (op === 0x30 || (S[--sp] === 0) === (op === 0x31)) pc = next + rel;
        break;
      }
      case 0x33: if (csp === 64) return end(7); C[csp++] = next; pc = prog.readUInt16LE(at); break; // CALL
      case 0x34: if (csp === 0) return end(8); pc = C[--csp]; break; // RET
      case 0x40: { const a = S[sp - 1]; if (a < 0 || a > 255) return end(6); S[sp - 1] = M[a]; break; } // LOAD
      case 0x41: { const a = S[sp - 1]; if (a < 0 || a > 255) return end(6); M[a] = S[sp - 2]; sp -= 2; break; } // STORE ( v addr -- )
      case 0x50: { // PRINT in decimal ASCII
        const v = S[--sp];
        let x = v < 0 ? -v : v; // -(-2^31) is 2^31 as a double
        const dig = [];
        do { dig.push(48 + (x % 10)); x = Math.floor(x / 10); } while (x > 0);
        if (v < 0) dig.push(45);
        if (olen + dig.length > 65_536) return end(11); // writes nothing
        for (let i = dig.length - 1; i >= 0; i--) out[olen++] = dig[i];
        break;
      }
      case 0x51: if (olen === 65_536) return end(11); out[olen++] = S[--sp] & 255; break; // PUTC, the low byte
      case 0x52: S[sp++] = ip < inp.length ? inp[ip++] : -1; break; // IN, -1 once used up
      case 0x53: return end(S[--sp] & 255); // EXIT, the low byte
      default: { // the two-operand group: b is the top, a below it
        const b = S[sp - 1];
        const a = S[sp - 2];
        if (b === 0 && op >= 0x13 && op <= 0x16) return end(4); // a division by zero
        let r;
        switch (op) {
          case 0x10: r = a + b; break; // ADD
          case 0x11: r = a - b; break; // SUB
          case 0x12: r = Math.imul(a, b); break; // MUL, exact low 32 bits
          case 0x13: r = a === -2147483648 && b === -1 ? a : (a / b) | 0; break; // DIV
          case 0x14: r = b === -1 ? 0 : a % b; break; // MOD
          case 0x15: r = Math.floor((a >>> 0) / (b >>> 0)); break; // DIVU
          case 0x16: r = (a >>> 0) % (b >>> 0); break; // MODU
          case 0x18: r = a & b; break; // AND
          case 0x19: r = a | b; break; // OR
          case 0x1a: r = a ^ b; break; // XOR
          case 0x1c: r = a << (b & 31); break; // SHL
          case 0x1d: r = a >> (b & 31); break; // SHR
          case 0x1e: r = a >>> (b & 31); break; // SHRU
          case 0x20: r = a === b ? 1 : 0; break; // EQ
          case 0x21: r = a < b ? 1 : 0; break; // LT
          default: r = (a >>> 0) < (b >>> 0) ? 1 : 0; break; // LTU (0x22)
        }
        S[sp - 2] = r;
        sp--;
        break;
      }
    }
  }
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  const [bytes, code] = run(readFileSync(join(I, id, 'program.bin')), readFileSync(join(I, id, 'input.bin')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), bytes);
  writeFileSync(join(O, id, 'exit.txt'), code + '\n');
}
