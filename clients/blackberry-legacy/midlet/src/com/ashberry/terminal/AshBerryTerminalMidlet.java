package com.ashberry.terminal;

import javax.microedition.lcdui.Alert;
import javax.microedition.lcdui.AlertType;
import javax.microedition.lcdui.Choice;
import javax.microedition.lcdui.Command;
import javax.microedition.lcdui.CommandListener;
import javax.microedition.lcdui.Display;
import javax.microedition.lcdui.Displayable;
import javax.microedition.lcdui.Form;
import javax.microedition.lcdui.List;
import javax.microedition.lcdui.StringItem;
import javax.microedition.lcdui.TextField;
import javax.microedition.midlet.MIDlet;
import javax.microedition.midlet.MIDletStateChangeException;

public final class AshBerryTerminalMidlet extends MIDlet implements CommandListener {
    private final Command backCommand = new Command("Back", Command.BACK, 1);
    private final Command exitCommand = new Command("Exit", Command.EXIT, 1);
    private final Command refreshCommand = new Command("Refresh", Command.SCREEN, 1);
    private final Command saveCommand = new Command("Save", Command.OK, 1);

    private Display display;
    private List home;
    private Form statusForm;
    private Form settingsForm;
    private TextField baseUrlField;
    private String baseUrl;

    protected void startApp() throws MIDletStateChangeException {
        display = Display.getDisplay(this);
        baseUrl = SettingsStore.loadBaseUrl();
        showHome();
    }

    protected void pauseApp() {
    }

    protected void destroyApp(boolean unconditional) throws MIDletStateChangeException {
    }

    private void showHome() {
        home = new List("ASHBERRY TERMINAL", Choice.IMPLICIT);
        home.append("[1] STATUS", null);
        home.append("[2] PRICE PHOENIX", null);
        home.append("[3] RECENT RUNS", null);
        home.append("[4] SETTINGS", null);
        home.append("[5] ABOUT", null);
        home.append("[6] EXIT", null);
        home.addCommand(exitCommand);
        home.setCommandListener(this);
        display.setCurrent(home);
    }

    private void showStatus() {
        statusForm = new Form("STATUS");
        statusForm.append(new StringItem(null, "Checking backend...\n"));
        statusForm.addCommand(backCommand);
        statusForm.addCommand(refreshCommand);
        statusForm.setCommandListener(this);
        display.setCurrent(statusForm);
        refreshStatus();
    }

    private void refreshStatus() {
        final String pingUrl = baseUrl + "/api/bb/ping";
        final String statusUrl = baseUrl + "/api/bb/model-status";
        new Thread(new Runnable() {
            public void run() {
                try {
                    String ping = HttpClient.get(pingUrl);
                    String status = HttpClient.get(statusUrl);
                    final String rendered = "ONLINE\n\n"
                        + ResponseParser.compact(ping)
                        + "\n"
                        + ResponseParser.compact(status);
                    display.callSerially(new Runnable() {
                        public void run() {
                            statusForm.deleteAll();
                            statusForm.append(new StringItem(null, rendered));
                        }
                    });
                } catch (final Exception exc) {
                    display.callSerially(new Runnable() {
                        public void run() {
                            statusForm.deleteAll();
                            statusForm.append(new StringItem(null, "OFFLINE\n\n" + exc.getMessage()));
                        }
                    });
                }
            }
        }).start();
    }

    private void showSettings() {
        settingsForm = new Form("SETTINGS");
        baseUrlField = new TextField("Base URL", baseUrl, 128, TextField.URL);
        settingsForm.append(baseUrlField);
        settingsForm.addCommand(saveCommand);
        settingsForm.addCommand(backCommand);
        settingsForm.setCommandListener(this);
        display.setCurrent(settingsForm);
    }

    private void showAbout() {
        Form about = new Form("ABOUT");
        about.append("AshBerry Terminal\n");
        about.append("Java ME native thin client.\n\n");
        about.append("Backend does all pricing, model, scenario, and storage work.\n");
        about.addCommand(backCommand);
        about.setCommandListener(this);
        display.setCurrent(about);
    }

    private void showNotImplemented(String label) {
        Alert alert = new Alert("NEXT PHASE", label + " is not implemented in this spike.", null, AlertType.INFO);
        alert.setTimeout(Alert.FOREVER);
        display.setCurrent(alert, home);
    }

    private void saveSettings() {
        String value = baseUrlField.getString();
        if (value == null || value.length() == 0) {
            showError("Base URL is required.");
            return;
        }
        baseUrl = value;
        SettingsStore.saveBaseUrl(baseUrl);
        showHome();
    }

    private void showError(String message) {
        Alert alert = new Alert("ERROR", message, null, AlertType.ERROR);
        alert.setTimeout(Alert.FOREVER);
        display.setCurrent(alert);
    }

    public void commandAction(Command command, Displayable displayable) {
        if (command == exitCommand) {
            notifyDestroyed();
            return;
        }
        if (command == backCommand) {
            showHome();
            return;
        }
        if (command == refreshCommand) {
            refreshStatus();
            return;
        }
        if (command == saveCommand) {
            saveSettings();
            return;
        }
        if (displayable == home && command == List.SELECT_COMMAND) {
            int selected = home.getSelectedIndex();
            if (selected == 0) {
                showStatus();
            } else if (selected == 1) {
                showNotImplemented("Phoenix pricing");
            } else if (selected == 2) {
                showNotImplemented("Recent runs");
            } else if (selected == 3) {
                showSettings();
            } else if (selected == 4) {
                showAbout();
            } else if (selected == 5) {
                notifyDestroyed();
            }
        }
    }
}
