# Java ME MIDlet Build Notes

This MIDlet source targets a Java ME / MIDP 2.0 style build, not the
BlackBerry-specific `net.rim.*` Java APIs.

Local status:

- `java` and `javac` are installed.
- `preverifier` is not installed.
- `emulator` is not installed.

That means this checkout has source code only. No `.jar`, `.jad`, or `.cod`
artifact has been built or installed from this folder.

Expected build flow once a Java ME SDK or compatible Wireless Toolkit exists:

1. Compile the sources against CLDC 1.1 / MIDP 2.0 APIs.
2. Preverify the compiled classes.
3. Package `AshBerryTerminal.jar`.
4. Generate a real `.jad` from `AshBerryTerminal.jad.template` with the actual
   JAR size.
5. Install the `.jad` / `.jar` pair or convert through the chosen BlackBerry
   deployment toolchain.

Do not commit generated `.jar`, `.jad`, `.cod`, or simulator output.
