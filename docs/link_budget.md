\# Link budget (all arithmetic derived here, never inline)

Noise:    −174 dBm/Hz + 10·log₁₀(1e6 Hz) + 7 dB NF = −174 + 60 + 7 = −107 dBm = 10^(−13.7) W

BS power: 30 dBm = 1.0 W

Path losses (d = distance):

&#x20; BS→user direct (NLoS UMi): 128.1 + 37.6 log10(d\_km)

&#x20; Cascade two-hop (Wu-Zhang): 61.4 + 35.8 log10(d\_m) with d = max(d\_BS→RIS, d\_RIS→user)

Geometry (env.yaml): d\_BS→RIS = 33.0 m → cascade floor ≈ 61.4 + 35.8·log10(33) ≈ 116 dB

d\_RIS→user ∈ \[d\_RIS, d\_RIS+\~50] m; users in \[45,60]×\[−10,10] ⇒ cascade PL ≈ 117–120 dB

Direct NW at \~55 m: 128.1 + 37.6·log10(0.055) ≈ 80 dB ⇒ direct is much stronger OR walls

attenuate it. Clé: G1 measures the actual margins — if random − no-RIS is thin, the ONLY allowed

knobs are (a) user-zone position, (b) p\_tx\_dbm, (c) explicit direct-blockage attenuation factor.

Never touch the PL formula constants without a design-decision entry (DD-011...).

