# Quasi-Stellar Object (QSO, quasar)

Can SDSS quasar data help to find quasars in LAMOST?

## Data

Data should be in the `data/` directory.

## Sloan Digital Sky Survey Data Release 14

Scope of the SDSS DR14: https://www.sdss.org/dr14/scope/.

The optical spectra catalog data `specObj-dr14.fits`:
https://www.sdss.org/dr14/data_access/bulk/#OpticalSpectraCatalogData
and description:
https://data.sdss.org/datamodel/files/SPECTRO_REDUX/specObj.html
or https://www.sdss.org/dr14/spectro/catalogs/.

### Sloan Digital Sky Survey Quasar Catalog: Fourteenth Data Release

`DR14Q_v4_4.fits` is the SDSS14Q catalog:
https://www.sdss.org/dr14/algorithms/qso_catalog/.

### Notes

The FITS file `5305/spec-5305-55984-0563.fits` is not the lite version
but the full version because there it no lite file available.

Lite version: https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/lite/PLATE4/spec.html
where downloaded using instruction on:
https://www.sdss.org/dr14/data_access/bulk/#OpticalSpectraPer-ObjectLiteFiles

Full version: https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html

### References

- Understanding the Optical Data: https://www.sdss.org/dr14/spectro/spectro_basics/
- Optical Spectra Data Quality Flags: https://www.sdss.org/dr14/spectro/quality/

## LAMOST Data Release 5 v3

LAMOST DR5 v3: http://dr5.lamost.org/.

The general catalog `dr5_v3.csv` is available on:
http://dr5.lamost.org/catalogue.

### LAMOST Quasar Catalog

LAMOST QSO catalogs are described on:
http://explore.china-vo.org/article/20190107155838.

The main file is `dr4dr5_qso_massflagged_v04.fits` of DR4 and DR5.

Other files are:

- `dr1_knownquasars_v01.fits`: DR1 known quasars.
- `dr1_newquasars_v01.fits`: DR1 new quasars.
- `dr23_table_v01.fits`: DR2 and DR3 quasars.
- `lamost_phase1_v02.fits`: full catalog.
