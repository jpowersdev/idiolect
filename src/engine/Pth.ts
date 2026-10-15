/**
 * Read a PyTorch state dict saved with torch.save: an uncompressed zip holding a pickle of
 * tensor descriptions and one entry of raw bytes per storage. Interprets only what state dicts
 * use, so no Python is needed to read VoxCPM2's codec weights.
 */

export interface RawTensor {
  readonly data: Float32Array | Int32Array
  readonly shape: ReadonlyArray<number>
}

/** The entries of an uncompressed (stored) zip archive, by name. */
const readZip = (bytes: Uint8Array) => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  // The end-of-central-directory record is within the last 64 KiB plus its own 22 bytes.
  let end = -1
  for (let at = bytes.length - 22; at >= Math.max(0, bytes.length - 65557); at--) {
    if (view.getUint32(at, true) === 0x06054b50) {
      end = at
      break
    }
  }
  if (end < 0) throw new Error("Not a zip archive.")
  let entries = view.getUint16(end + 10, true)
  let at = view.getUint32(end + 16, true)
  // Zip64 archives keep the real counts in their own end record.
  if (entries === 0xffff || at === 0xffffffff) {
    const locator = end - 20
    const record = Number(view.getBigUint64(locator + 8, true))
    entries = Number(view.getBigUint64(record + 32, true))
    at = Number(view.getBigUint64(record + 48, true))
  }
  const files = new Map<string, Uint8Array>()
  for (let i = 0; i < entries; i++) {
    if (view.getUint32(at, true) !== 0x02014b50) throw new Error("Corrupt zip directory.")
    const method = view.getUint16(at + 10, true)
    let size = view.getUint32(at + 20, true)
    const nameLength = view.getUint16(at + 28, true)
    const extraLength = view.getUint16(at + 30, true)
    const commentLength = view.getUint16(at + 32, true)
    let local = view.getUint32(at + 42, true)
    const name = new TextDecoder().decode(bytes.subarray(at + 46, at + 46 + nameLength))
    // Zip64 extra field: 64-bit sizes and offset, for the fields saturated above.
    for (let e = at + 46 + nameLength; e < at + 46 + nameLength + extraLength;) {
      const id = view.getUint16(e, true)
      const length = view.getUint16(e + 2, true)
      if (id === 0x0001) {
        let field = e + 4
        if (view.getUint32(at + 24, true) === 0xffffffff) field += 8
        if (size === 0xffffffff) {
          size = Number(view.getBigUint64(field, true))
          field += 8
        }
        if (local === 0xffffffff) local = Number(view.getBigUint64(field, true))
      }
      e += 4 + length
    }
    if (method !== 0) throw new Error(`Zip entry ${name} is compressed; torch.save archives are not.`)
    const data = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true)
    files.set(name, bytes.subarray(data, data + size))
    at += 46 + nameLength + extraLength + commentLength
  }
  return files
}

type Value = unknown
interface Global {
  readonly module: string
  readonly name: string
}
interface Storage {
  readonly kind: string
  readonly key: string
}
const MARK = Symbol("mark")

/** Unpickle a state dict: protocol 2 opcodes, with tensors rebuilt from their storages. */
const unpickle = (pickle: Uint8Array, storage: (s: Storage) => Uint8Array) => {
  const view = new DataView(pickle.buffer, pickle.byteOffset, pickle.byteLength)
  const stack: Array<Value> = []
  const memo = new Map<number, Value>()
  let at = 0
  const popMark = () => {
    const index = stack.lastIndexOf(MARK)
    const items = stack.splice(index)
    items.shift()
    return items
  }
  const string = (length: number) => {
    const text = new TextDecoder().decode(pickle.subarray(at, at + length))
    at += length
    return text
  }
  const line = () => {
    const stop = pickle.indexOf(0x0a, at)
    const text = new TextDecoder().decode(pickle.subarray(at, stop))
    at = stop + 1
    return text
  }
  const rebuild = (args: Array<Value>): RawTensor => {
    const [store, offset, size, stride] = args as [Storage, number, Array<number>, Array<number>]
    const bytes = storage(store)
    const isInt = store.kind === "IntStorage"
    const source = isInt
      ? new Int32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
      : new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
    const count = size.reduce((a, b) => a * b, 1)
    const data = isInt ? new Int32Array(count) : new Float32Array(count)
    // Copy in row-major order through the strides, so views and offsets read correctly.
    const index = new Array<number>(size.length).fill(0)
    for (let i = 0; i < count; i++) {
      let position = offset
      for (let d = 0; d < size.length; d++) position += index[d]! * stride[d]!
      data[i] = source[position]!
      for (let d = size.length - 1; d >= 0; d--) {
        if (++index[d]! < size[d]!) break
        index[d] = 0
      }
    }
    return { data, shape: size }
  }
  while (at < pickle.length) {
    const op = pickle[at++]!
    switch (op) {
      case 0x80: at++; break // PROTO
      case 0x95: at += 8; break // FRAME
      case 0x7d: stack.push(new Map()); break // EMPTY_DICT
      case 0x29: stack.push([]); break // EMPTY_TUPLE
      case 0x28: stack.push(MARK); break // MARK
      case 0x71: memo.set(pickle[at++]!, stack.at(-1)); break // BINPUT
      case 0x72: memo.set(view.getUint32(at, true), stack.at(-1)); at += 4; break // LONG_BINPUT
      case 0x94: memo.set(memo.size, stack.at(-1)); break // MEMOIZE
      case 0x68: stack.push(memo.get(pickle[at++]!)); break // BINGET
      case 0x6a: stack.push(memo.get(view.getUint32(at, true))); at += 4; break // LONG_BINGET
      case 0x58: { const n = view.getUint32(at, true); at += 4; stack.push(string(n)); break } // BINUNICODE
      case 0x8c: { const n = pickle[at++]!; stack.push(string(n)); break } // SHORT_BINUNICODE
      case 0x4b: stack.push(pickle[at++]!); break // BININT1
      case 0x4d: stack.push(view.getUint16(at, true)); at += 2; break // BININT2
      case 0x4a: stack.push(view.getInt32(at, true)); at += 4; break // BININT
      case 0x88: stack.push(true); break // NEWTRUE
      case 0x89: stack.push(false); break // NEWFALSE
      case 0x4e: stack.push(null); break // NONE
      case 0x85: stack.push([stack.pop()]); break // TUPLE1
      case 0x86: { const b = stack.pop(); const a = stack.pop(); stack.push([a, b]); break } // TUPLE2
      case 0x87: { const c = stack.pop(); const b = stack.pop(); const a = stack.pop(); stack.push([a, b, c]); break } // TUPLE3
      case 0x74: stack.push(popMark()); break // TUPLE
      case 0x63: stack.push({ module: line(), name: line() } satisfies Global); break // GLOBAL
      case 0x93: { const name = stack.pop() as string; const module = stack.pop() as string; stack.push({ module, name } satisfies Global); break } // STACK_GLOBAL
      case 0x51: { // BINPERSID: ("storage", type, key, location, count)
        const [, kind, key] = stack.pop() as [string, Global, string]
        stack.push({ kind: kind.name, key } satisfies Storage)
        break
      }
      case 0x52: { // REDUCE
        const args = stack.pop() as Array<Value>
        const callable = stack.pop() as Global
        if (callable.name === "_rebuild_tensor_v2" || callable.name === "_rebuild_tensor") stack.push(rebuild(args))
        else if (callable.name === "OrderedDict") stack.push(new Map())
        else throw new Error(`Unsupported pickle callable ${callable.module}.${callable.name}.`)
        break
      }
      case 0x73: { const value = stack.pop(); const key = stack.pop(); (stack.at(-1) as Map<Value, Value>).set(key, value); break } // SETITEM
      case 0x75: { // SETITEMS
        const items = popMark()
        const dict = stack.at(-1) as Map<Value, Value>
        for (let i = 0; i < items.length; i += 2) dict.set(items[i], items[i + 1])
        break
      }
      case 0x62: stack.pop(); break // BUILD: state for OrderedDict metadata, unused
      case 0x2e: return stack.pop() // STOP
      default: throw new Error(`Unsupported pickle opcode 0x${op.toString(16)}.`)
    }
  }
  throw new Error("The pickle ended without STOP.")
}

/** The tensors of a torch.save state dict, by name. A wrapping {"state_dict": …} is unwrapped. */
export const readStateDict = (bytes: Uint8Array): Map<string, RawTensor> => {
  const files = readZip(bytes)
  const pickleName = [...files.keys()].find((name) => name.endsWith("/data.pkl"))
  if (!pickleName) throw new Error("No data.pkl in the archive; is this a torch.save file?")
  const root = pickleName.slice(0, -"data.pkl".length)
  let state = unpickle(files.get(pickleName)!, ({ key }) => {
    const data = files.get(`${root}data/${key}`)
    if (!data) throw new Error(`Missing storage ${key}.`)
    return data
  }) as Map<string, unknown>
  if (state.has("state_dict")) state = state.get("state_dict") as Map<string, unknown>
  return state as Map<string, RawTensor>
}

/**
 * Fold weight normalization into plain weights: each `x.weight_g` and `x.weight_v` pair becomes
 * `x.weight = g · v / ‖v‖`, with one norm per output channel (weight_norm's default dim 0).
 */
export const foldWeightNorm = (tensors: Map<string, RawTensor>) => {
  const out = new Map<string, RawTensor>()
  for (const [name, tensor] of tensors) {
    if (name.endsWith(".weight_g")) continue
    if (!name.endsWith(".weight_v")) {
      out.set(name, tensor)
      continue
    }
    const base = name.slice(0, -".weight_v".length)
    const g = tensors.get(`${base}.weight_g`)!.data
    const v = tensor.data as Float32Array
    const channels = tensor.shape[0]!
    const per = v.length / channels
    const weight = new Float32Array(v.length)
    for (let c = 0; c < channels; c++) {
      let sum = 0
      for (let i = c * per; i < (c + 1) * per; i++) sum += v[i]! * v[i]!
      const scale = g[c]! / Math.sqrt(sum)
      for (let i = c * per; i < (c + 1) * per; i++) weight[i] = v[i]! * scale
    }
    out.set(`${base}.weight`, { data: weight, shape: tensor.shape })
  }
  return out
}

/**
 * The tensors as a safetensors file, which effect-torch loads straight into GPU memory: far
 * leaner than building each tensor from a host array.
 */
export const toSafetensors = (tensors: Map<string, RawTensor>) => {
  const header: Record<string, { dtype: string; shape: ReadonlyArray<number>; data_offsets: [number, number] }> = {}
  let offset = 0
  for (const [name, { data, shape }] of tensors) {
    header[name] = { dtype: data instanceof Float32Array ? "F32" : "I32", shape, data_offsets: [offset, offset + data.byteLength] }
    offset += data.byteLength
  }
  let json = JSON.stringify(header)
  // The data starts 8-byte aligned.
  json += " ".repeat((8 - ((8 + json.length) % 8)) % 8)
  const encoded = new TextEncoder().encode(json)
  const bytes = new Uint8Array(8 + encoded.length + offset)
  new DataView(bytes.buffer).setBigUint64(0, BigInt(encoded.length), true)
  bytes.set(encoded, 8)
  let at = 8 + encoded.length
  for (const { data } of tensors.values()) {
    bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), at)
    at += data.byteLength
  }
  return bytes
}
