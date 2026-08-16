# Dataset provenance

The five processed multi-view matrices are third-party. Their source, rights
note, view order, and SHA-256 checksum are recorded here so an authorized copy
can be verified. The revision does not redistribute these matrices under a new
blanket license, and it reconstructs all graphs from the normalized feature
matrices without labels.

## View order (solver convention, fixed in `code/dataset_config.m`)

- UCI: `X3` KAR (64), `X2` FOU (76), `X1` FAC (216)
- BBCSport: `X1` (3183), `X2` (3203)
- Yale/ORL: `X3` Gabor (6750), `X2` LBP (3304), `X1` intensity (4096)
- Scene-15: `X3` CENTRIST (1240), `X2` PRI-CoLBP (1180), `X1` PHOW (1800)

## Source, rights, and checksums

| Dataset | Samples / classes | Source and rights note | Local file | SHA-256 |
|---|---:|---|---|---|
| UCI Multiple Features | 2000 / 10 | UCI record <https://doi.org/10.24432/C5HC70> (CC BY 4.0). The official collection has six views; this experiment uses the three-view subset above. | `datasets/uci.mat` | `FB0539E42E4A7BFFC1AAA0D7392438CA2BC62B391673DAB8222D9B944457F56F` |
| BBCSport | 544 / 5 | Greene & Cunningham corpus, DOI <https://doi.org/10.1145/1143844.1143892>. BBC owns the underlying text: do not redistribute unless the terms for this processed copy are confirmed; otherwise use the checksum plus the acquisition note below. | `datasets/bbcsport_2view.mat` | `7508329E81D33950A494375DDDE2E8010E73CB0C9C09AC2B807752AB19F83FB3` |
| Yale Face Database | 165 / 15 | <https://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.html>. The local file is a processed multi-view feature matrix, not the raw GIF release. | `datasets/yale.mat` | `C7C080BBAF091066891F567EE5194CE8D8BFC959FB9E5A586B7123ACF93586BA` |
| ORL (AT&T) | 400 / 40 | <https://cam-orl.co.uk/facedatabase.html/> (AT&T Laboratories Cambridge). The local file contains processed descriptors, not the raw PGM images. | `datasets/ORL.mat` | `F459A43245B1EBD0E15F301707C3E3C371FF66B56279C8F2B6D0EC8098CC522D` |
| Scene-15 | 4485 / 15 | Lazebnik, Schmid, Ponce, DOI <https://doi.org/10.1109/CVPR.2006.68>. The processed descriptor configuration follows published multi-view clustering benchmarks. | `datasets/scene15.mat` | `92A905A388925AFCE9498DD26525FC2CEB0B6B4528251137D33DF89453AC8CD5` |

## Acquisition / preparation

- The files above are processed feature-by-sample matrices, not the raw
  releases. To reproduce a hash, obtain the source from the link above, extract
  the same view subset in the order fixed by `code/dataset_config.m`, and
  column-normalize before graph construction (as the revision workflow does).
- Do not substitute a same-named package with different views. Compare each
  `.mat` against the hash above before a run.
- The revision does not use any legacy precomputed `*_A.mat` graph.
