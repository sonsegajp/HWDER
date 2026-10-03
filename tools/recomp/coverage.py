"""Translate every word in a module's .text and report unimplemented encodings."""
import sys, collections, capstone
sys.path.insert(0, __file__.rsplit('recomp', 1)[0])
from nso import Process
from recomp.a64 import Translator, Unimplemented

def main(module='main'):
    p = Process('../data/exefs')
    m = [x for x in p.modules if x.name == module][0]
    tr = Translator()
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    fails = collections.Counter(); examples = {}
    total = 0
    import struct
    for off in range(m.text_off, m.text_off + m.text_size, 4):
        w = struct.unpack_from('<I', m.image, off)[0]
        total += 1
        try:
            tr.translate(w, m.base + off)
        except Unimplemented as e:
            k = str(e); fails[k] += 1
            examples.setdefault(k, []).append((m.base + off, w))
        except Exception as e:
            k = 'CRASH ' + type(e).__name__ + ' ' + str(e); fails[k] += 1
            examples.setdefault(k, []).append((m.base + off, w))
    print(f"{total} words, {sum(fails.values())} untranslated")
    for k, v in fails.most_common():
        ex = examples[k][:4]
        dis = []
        for a, w in ex:
            r = list(md.disasm_lite(struct.pack('<I', w), a))
            dis.append(f"{w:08x} {r[0][2]} {r[0][3]}" if r else f"{w:08x} ?")
        print(f"{v:7d} {k:40s} | " + " ; ".join(dis))

if __name__ == '__main__':
    main(*sys.argv[1:])
