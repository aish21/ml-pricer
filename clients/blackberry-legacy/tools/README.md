# Legacy BlackBerry Tools

This folder is a placeholder for notes or small helper scripts related to the
Java ME / BlackBerry legacy build workflow.

Do not commit vendor SDKs, generated `.cod` files, signing keys, Desktop
Software installers, or machine-specific paths.

Useful commands to verify after installing the toolchain:

```powershell
java -version
javac -version
preverifier
emulator
```

If later converting/installing through BlackBerry-specific tools, also verify:

```powershell
rapc
javaloader.exe -h
```
