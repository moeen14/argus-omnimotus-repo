# Upload and release preparation

Upload the **contents of `Argus-Code`** as the code repository root. Keep `README.md`, `.github`, `.gitignore`, `hardware`, `firmware`, `applications`, `tuning`, `vision` and `deployment` at that level. Include hidden `.github`, `.gitignore` and `.gitattributes` files. The parent publication workspace and original export do not belong in this repository.

Use Git or GitHub Desktop so `.gitignore` excludes local configuration, caches, model binaries, recordings and environments. Browser upload does not apply `.gitignore` filtering: select only intended source files if using the browser.

The repository is organized for upload, with these release details still owned by the maintainer:

- Choose and add the software `LICENSE`; separately select hardware/model/dataset licenses.
- Fill `project-links.json` with the actual homepage, code, hardware, model, dataset and paper URLs when available.
- Run `python tools/update_links.py` to refresh the main and hardware resource pages from the root link configuration. `--strict` additionally rejects missing URLs.
- Add the public repository URL to `CITATION.cff` as `repository-code` when known. Update the manuscript citation when a DOI or published record exists.
- Keep CAD, assembly drawings, circuit files and the BOM under `hardware/`. Complete printable exports, circuit drawings and the purchasing BOM as they become available.
- Publish the ONNX models and datasets with appropriate cards and then verify downloads against the manifest.

The website remains a separate repository. Edit `Project-Homepage` in the local workspace; refresh `Website-Publish` only when the project owner requests the finalized export.

Before a tagged robot release, record end-to-end Jetson/robot validation, calibrated geometry and model hashes. CI checks syntax, tools and firmware compilation; it does not certify robot motion.

## Large CAD files

Run `git lfs install` before `git add`. The SolidWorks master file is about 124 MB and must be stored with Git LFS. Uppercase and lowercase SolidWorks extensions are covered by `hardware/.gitattributes`. Verify tracking with `git lfs ls-files` after staging. The hardware BOM and drawing are explicitly allowed by the root `.gitignore`, while robot recordings remain excluded.
