# AshBerry Terminal Legacy Client

This folder contains a minimal BlackBerry OS 6 Java thin-client spike for the
ML-Pricer BlackBerry terminal.

The primary BlackBerry experience remains the server-rendered web terminal at
`/bb`. This sideloaded app is only a launcher shell: it opens the backend
terminal URL in the native BlackBerry browser.

## What It Does

- Shows a small `ASHBERRY TERMINAL` screen.
- Opens `DEFAULT_TERMINAL_URL` when ENTER is pressed.
- Also provides an `Open Terminal` menu item.

## What It Does Not Do

- It does not price products locally.
- It does not contain payoff formulas.
- It does not load model/scaler artifacts.
- It does not run scenario logic.
- It does not store API keys or secrets.

## Source

```text
src/com/ashberry/terminal/AshBerryTerminal.java
```

Before building for a real device, update the placeholder URL:

```java
static final String DEFAULT_TERMINAL_URL = "http://192.168.1.100:8000/bb";
```

Use the PC's LAN IP and the backend port, for example:

```text
http://<PC_LOCAL_IP>:8000/bb
```

Do not commit machine-specific IP addresses or secrets.

## Tooling Status

This spike was added without a local BlackBerry build toolchain available on
PATH. The following commands were not found locally:

- `java`
- `javac`
- `rapc`
- `javaloader`
- `bbwp`

That means the Java source has not yet been compiled into a `.cod` file and has
not yet been sideloaded from this checkout.

## Expected Legacy Toolchain

For a BlackBerry Bold 9780 / BlackBerry OS 6 device, expect to need:

- BlackBerry Java SDK or BlackBerry JDE / Eclipse plugin from the legacy era.
- A compatible Java JDK required by that BlackBerry SDK.
- BlackBerry Desktop Software or USB drivers.
- `javaloader.exe` for command-line sideloading, if using the CLI path.

The exact SDK download and Java version need to be confirmed on the build
machine because the legacy toolchain is old and availability varies.

## Build

The source uses BlackBerry-specific `net.rim.*` APIs and cannot be compiled
with stock `javac` alone.

Expected options:

1. Import `src/com/ashberry/terminal/AshBerryTerminal.java` into a BlackBerry
   Java SDK/JDE project.
2. Set the application entry point to:

```text
com.ashberry.terminal.AshBerryTerminal
```

3. Build/sign as required by the installed SDK.

If using legacy command-line tools, the eventual build will likely involve
`rapc`, but the exact command should be captured after the SDK is installed and
verified.

## Sideload

Once a `.cod` file exists, expected CLI loading shape:

```powershell
javaloader.exe load AshBerryTerminal.cod
```

Alternative:

- Use BlackBerry Desktop Software to install the generated app package.

## Runtime Flow

1. Start the backend:

```powershell
python -m uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000
```

2. Confirm the BlackBerry browser can open:

```text
http://<PC_LOCAL_IP>:8000/bb
```

3. Build and sideload this launcher.
4. Open `AshBerry Terminal` from the BlackBerry home screen.
5. Press ENTER to open the terminal page.

## Phase 7B Candidates

- Add a small settings screen for host and port.
- Persist the configured backend URL locally.
- Evaluate BrowserField embedding after the basic launcher builds.
- Add a backend reachability check if it proves useful.
