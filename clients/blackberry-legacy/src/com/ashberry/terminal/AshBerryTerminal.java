package com.ashberry.terminal;

import net.rim.blackberry.api.browser.Browser;
import net.rim.blackberry.api.browser.BrowserSession;
import net.rim.device.api.system.Characters;
import net.rim.device.api.ui.Field;
import net.rim.device.api.ui.MenuItem;
import net.rim.device.api.ui.UiApplication;
import net.rim.device.api.ui.component.Menu;
import net.rim.device.api.ui.component.RichTextField;
import net.rim.device.api.ui.container.MainScreen;

/**
 * Minimal BlackBerry OS 6 thin-client launcher for the server-rendered terminal.
 *
 * This app intentionally contains no pricing logic, model artifacts, secrets,
 * or product-specific rules. The FastAPI backend remains the source of truth.
 */
public final class AshBerryTerminal extends UiApplication {
    static final String DEFAULT_TERMINAL_URL = "http://192.168.1.100:8000/bb";

    public static void main(String[] args) {
        AshBerryTerminal app = new AshBerryTerminal();
        app.enterEventDispatcher();
    }

    private AshBerryTerminal() {
        pushScreen(new TerminalHomeScreen());
    }

    private static final class TerminalHomeScreen extends MainScreen {
        private TerminalHomeScreen() {
            setTitle("ASHBERRY TERMINAL");
            add(new RichTextField(
                "Legacy client online.\n\n" +
                "Press ENTER to open terminal.\n\n" +
                "URL:\n" + DEFAULT_TERMINAL_URL,
                Field.NON_FOCUSABLE
            ));
        }

        protected boolean keyChar(char key, int status, int time) {
            if (key == Characters.ENTER) {
                openTerminal();
                return true;
            }
            return super.keyChar(key, status, time);
        }

        protected void makeMenu(Menu menu, int instance) {
            menu.add(new MenuItem("Open Terminal", 10, 10) {
                public void run() {
                    openTerminal();
                }
            });
            super.makeMenu(menu, instance);
        }

        private void openTerminal() {
            BrowserSession session = Browser.getDefaultSession();
            session.displayPage(DEFAULT_TERMINAL_URL);
            session.showBrowser();
        }
    }
}
