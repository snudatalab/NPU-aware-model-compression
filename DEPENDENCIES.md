# Dependencies

## Public packages

The reference environment uses Python 3.10, PyTorch 2.8.0 CPU wheels,
torchvision 0.23.0 CPU wheels, and NumPy. Install them with:

```bash
pip install -r requirements.txt
```

## RBLN SDK

Compilation and NPU execution require the vendor-provided `rebel-compiler`
package and a compatible RBLN driver/firmware installation. The package is not
included in `requirements.txt` because it requires an RBLN Portal account.

For the RBLN SDK 0.9.3 validation environment:

```bash
pip install --extra-index-url https://pypi.rbln.ai/simple \
  rebel-compiler==0.9.3.post1
```

Use the compiler version recommended for the target server's driver. Do not
commit portal credentials, package-index tokens, private wheels, or compiled
`.rbln` artifacts to the repository.
