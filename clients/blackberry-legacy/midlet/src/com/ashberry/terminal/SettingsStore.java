package com.ashberry.terminal;

import javax.microedition.rms.RecordStore;

public final class SettingsStore {
    public static final String DEFAULT_BASE_URL = "http://192.168.1.100:8000";
    private static final String STORE_NAME = "AshBerrySettings";

    private SettingsStore() {
    }

    public static String loadBaseUrl() {
        RecordStore store = null;
        try {
            store = RecordStore.openRecordStore(STORE_NAME, true);
            if (store.getNumRecords() == 0) {
                return DEFAULT_BASE_URL;
            }
            byte[] data = store.getRecord(1);
            return new String(data);
        } catch (Exception exc) {
            return DEFAULT_BASE_URL;
        } finally {
            close(store);
        }
    }

    public static void saveBaseUrl(String baseUrl) {
        RecordStore store = null;
        try {
            store = RecordStore.openRecordStore(STORE_NAME, true);
            byte[] data = baseUrl.getBytes();
            if (store.getNumRecords() == 0) {
                store.addRecord(data, 0, data.length);
            } else {
                store.setRecord(1, data, 0, data.length);
            }
        } catch (Exception exc) {
        } finally {
            close(store);
        }
    }

    private static void close(RecordStore store) {
        if (store == null) {
            return;
        }
        try {
            store.closeRecordStore();
        } catch (Exception exc) {
        }
    }
}
