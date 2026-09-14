# Contributing

Describe the hardware, firmware revisions, JetPack/TensorRT versions and reproducible steps when reporting an issue. Include relevant text logs without credentials or private device information.

Keep changes focused. Explain control behavior, calibration assumptions and compatibility changes in the pull request and documentation. Code comments are not required by this project; keep explanations in the documentation and preserve any required third-party license notices.

Run these hardware-independent checks:

```bash
python -m compileall -q applications tuning vision archive experiments deployment tools tests
python -m unittest discover -s tests -v
python tools/firmware.py compile
```

Never run the `tuning/*_test.py` programs as a unit-test suite: they access physical hardware. Record hardware validation separately, including motion limits, model hashes and observed failures. A passing compile or CI run is not physical validation.

Keep datasets, weights, TensorRT engines, virtual environments and generated recordings out of Git. Update the model manifest only with a documented model change. Keep hardware files under [`hardware/`](hardware/) and model metadata under [`models/`](models/).
