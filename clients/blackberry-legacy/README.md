# AshBerry Terminal Java ME Client

This folder contains the Java ME native thin-client spike for the ML-Pricer
BlackBerry terminal.

The BlackBerry app renders its own compact terminal UI and talks to FastAPI over
local HTTP. It does not open the BlackBerry browser for the main flow and does
not parse the `/bb` HTML pages.

The server-rendered `/bb` browser terminal remains useful as the proven fallback
and manual testing path.

## Architecture

```text
BlackBerry Bold 9780 Java ME MIDlet
  -> HTTP over local Wi-Fi
  -> FastAPI /api/bb/* plain-text endpoints
  -> pricing service / model cache / scenario service
  -> SQLite run store
  -> compact text response
  -> Java ME app renders result
```

## What It Does

- Shows a native `ASHBERRY TERMINAL` menu.
- Stores a configurable backend base URL in RMS.
- Calls `GET /api/bb/ping`.
- Calls `GET /api/bb/model-status`.
- Renders backend status inside the MIDlet.
- Shows settings and about screens.

## What It Does Not Do

- It does not price products locally.
- It does not contain payoff formulas.
- It does not load model/scaler artifacts.
- It does not run scenario logic.
- It does not store API keys or secrets.
- It does not implement native Phoenix pricing yet.

## Source

```text
midlet/
  src/com/ashberry/terminal/
    AshBerryTerminalMidlet.java
    HttpClient.java
    ResponseParser.java
    SettingsStore.java
  manifest/MANIFEST.MF
  AshBerryTerminal.jad.template
  build-notes.md
```

The default base URL placeholder is:

```text
http://192.168.1.100:8000
```

Use the MIDlet settings screen to change it on device, or change it locally
before building if needed:

```text
http://<PC_LOCAL_IP>:8000
```

Do not commit machine-specific IP addresses or secrets.

## Backend API

The MIDlet uses compact plain-text endpoints:

```text
GET /api/bb/ping
GET /api/bb/model-status
GET /api/bb/products
```

Example model-status response:

```text
OK
PHOENIX=READY,COLD
ACCUM=READY,COLD
BARRIER=READY,COLD
```

Plain text is intentional: Java ME has limited standard library support, so this
avoids a JSON dependency and keeps parsing simple.

## Tooling Status

Installed locally:

- `java`
- `javac`

Missing locally:

- `preverifier`
- `emulator`
- Java ME Wireless Toolkit / Java ME SDK
- BlackBerry install tooling

That means the MIDlet source has not been compiled into a `.jar`, converted to
`.cod`, or sideloaded from this checkout.

## Build

The source uses standard Java ME APIs:

- `javax.microedition.midlet.MIDlet`
- `javax.microedition.lcdui.*`
- `javax.microedition.io.HttpConnection`
- `javax.microedition.rms.*`

It intentionally avoids `net.rim.*` APIs for Track A.

Expected build flow after installing a Java ME SDK or compatible Wireless
Toolkit:

1. Compile against CLDC 1.1 / MIDP 2.0.
2. Preverify classes.
3. Package `AshBerryTerminal.jar`.
4. Generate a real `.jad` from `AshBerryTerminal.jad.template` with the actual
   JAR size.
5. Install the `.jad` / `.jar` pair or convert through the chosen BlackBerry
   deployment toolchain.

Do not commit generated `.jar`, `.jad`, `.cod`, or simulator output.

## Runtime Test Flow

1. Start the backend:

```powershell
python -m uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000
```

2. From a desktop browser, confirm:

```text
http://127.0.0.1:8000/api/bb/ping
http://127.0.0.1:8000/api/bb/model-status
```

3. From the BlackBerry browser, confirm:

```text
http://<PC_LOCAL_IP>:8000/api/bb/ping
```

4. Build and install the MIDlet after Java ME tooling is available.
5. Open `AshBerry Terminal`.
6. Set the backend base URL.
7. Open `STATUS` and confirm the model-status text renders.

## Phase 7B Candidates

- Install/verify Java ME build tooling.
- Produce and install a real `.jar` / `.jad` package.
- Add a Phoenix-only native pricing form.
- Add recent-runs view.
- Add compact response parsing with field labels and error handling polish.
