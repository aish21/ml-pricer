# Java ME MIDlet Build Notes

This MIDlet source targets a Java ME / MIDP 2.0 style build, not the
BlackBerry-specific `net.rim.*` Java APIs.

Local status:

- `java` and `javac` are installed.
- `preverifier` is not installed.
- `emulator` is not installed.

That means this checkout has source code and a build script, but no `.jar`,
`.jad`, or `.cod` artifact has been built or installed from this folder.

## Recommended Toolchain

Install Sun Java Wireless Toolkit 2.5.2_01, or a compatible Java ME SDK with:

- CLDC 1.1 API jar
- MIDP 2.0 API jar
- `preverify` or `preverifier`
- optional emulator

Prefer an installation path without spaces:

```text
C:\WTK252
```

Avoid paths like:

```text
C:\Program Files\WTK252
```

Old Java ME tools are fragile around spaces in paths.

After installation, set:

```powershell
$env:WTK_HOME="C:\WTK252"
```

The current machine has Java 8 installed, but not WTK/preverification tooling.

## Build Script

From this folder:

```powershell
.\build.ps1
```

The script performs:

1. Compile the sources against CLDC 1.1 / MIDP 2.0 APIs.
2. Preverify the compiled classes.
3. Package `AshBerryTerminal.jar`.
4. Generate a real `.jad` from `AshBerryTerminal.jad.template` with the actual
   JAR size.
5. Install the `.jad` / `.jar` pair or convert through the chosen BlackBerry
   deployment toolchain.

Generated files are written to:

```text
dist/AshBerryTerminal.jar
dist/AshBerryTerminal.jad
```

The script fails early with a clear error if `WTK_HOME`, API jars, or
preverification tools are missing.

If your installed `javac` rejects the default CLDC-style flags, install an older
JDK compatible with the Wireless Toolkit or override:

```powershell
$env:JAVAC_SOURCE="1.6"
$env:JAVAC_TARGET="1.6"
.\build.ps1
```

Use those overrides only as a tooling workaround; the eventual device build
should still be validated on the real BlackBerry.

## Install Option A: microSD Copy

1. Build the MIDlet.
2. Copy these files to the BlackBerry microSD card:

```text
dist/AshBerryTerminal.jar
dist/AshBerryTerminal.jad
```

3. On the BlackBerry, open File Manager.
4. Open the `.jad` first if possible; otherwise open the `.jar`.
5. Install and launch `AshBerry Terminal`.

## Install Option B: Local OTA-Style Download

After building, serve the `dist` folder from a local web server and open the
JAD URL from the BlackBerry browser.

Example concept:

```text
http://<PC_LOCAL_IP>:8010/AshBerryTerminal.jad
```

The server should return:

- `.jad`: `text/vnd.sun.j2me.app-descriptor`
- `.jar`: `application/java-archive`

If the BlackBerry refuses to install from a generic static server, use the
microSD option first and capture the exact error.

## Physical Test Checklist

1. Start the backend:

```powershell
python -m uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000
```

2. Confirm desktop can open:

```text
http://<PC_LOCAL_IP>:8000/api/bb/ping
```

3. Confirm the BlackBerry browser can open the same ping URL.
4. Install the MIDlet.
5. Open `AshBerry Terminal`.
6. Go to `SETTINGS`.
7. Set:

```text
http://<PC_LOCAL_IP>:8000
```

8. Go to `STATUS`.
9. Confirm the app displays `ONLINE`, `SERVICE=ASHBERRY`, and model statuses.
10. If it fails, record the exact error text shown by the MIDlet.

Do not commit generated `.jar`, `.jad`, `.cod`, or simulator output.
