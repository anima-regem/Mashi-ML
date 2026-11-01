"""Explicit sequence inventory. No splitting into connected components/codepoints."""
import unicodedata

BASE = list("അആഇഈഉഊഋഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറൺൻർൽൾൿ")
CONSONANTS = list("കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ")
VOWELS = list("ാിീുൂൃെേൈൊോൌ")
CONJUNCTS = "ക്ക ക്ത ക്ഷ ഗ്ഗ ഗ്ന ങ്ക ങ്ങ ച്ച ച്ഛ ച്ച്യ ജ്ജ ജ്ഞ ഞ്ച ഞ്ജ ഞ്ഞ ട്ട ണ്ട ണ്ണ ത്ത ത്ഥ ത്ര ത്മ ദ്ദ ദ്ധ ദ്ര ന്ത ന്ദ ന്ന ന്മ പ്പ പ്ത പ്ര പ്ല ബ്ബ ബ്ധ ബ്ര ഭ്ര മ്മ മ്പ മ്മ്യ യ്യ ല്ല ല്പ ല്മ വ്വ ശ്ശ ശ്ച ശ്ര സ്സ സ്ത സ്ന സ്മ ഹ്ന ള്ള റ്റ ന്റ".split()


def inventory():
    result = []
    seen = set()
    def add(sequence, tier):
        if sequence not in seen:
            seen.add(sequence)
            result.append({"sequence": sequence, "codepoints": [f"U+{ord(c):04X}" for c in sequence], "tier": tier,
                           "unicode_names": [unicodedata.name(c, "UNKNOWN") for c in sequence],
                           "label_policy": "Exact codepoint sequence; no NFC replacement or joiner deletion."})
    for s in BASE + list("൦൧൨൩൪൫൬൭൮൯"):
        add(s, "basic")
    for c in CONSONANTS:
        for sign in VOWELS:
            add(c + sign, "vowel-context")
        add(c + "്", "virama")
    for c in CONJUNCTS:
        for suffix in ["", "ാ", "ു"]:
            add(c + suffix, "conjunct")
    for c in "ണനരലള":
        add(c + "്\u200d", "legacy-joiner")
        add(c + "്\u200c", "explicit-nonjoiner")
    return result
