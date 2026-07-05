# Legacy BlackBerry Tools

This folder is a placeholder for notes or small helper scripts related to the
BlackBerry legacy build workflow.

Do not commit vendor SDKs, generated `.cod` files, signing keys, Desktop
Software installers, or machine-specific paths.

Useful commands to verify after installing the toolchain:

```powershell
java -version
javac -version
rapc
javaloader.exe -h
```

Expected sideload shape after a successful build:

```powershell
javaloader.exe load AshBerryTerminal.cod
```
