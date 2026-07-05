package com.ashberry.terminal;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.microedition.io.Connector;
import javax.microedition.io.HttpConnection;

public final class HttpClient {
    private HttpClient() {
    }

    public static String get(String url) throws IOException {
        HttpConnection connection = null;
        InputStream input = null;
        try {
            connection = (HttpConnection) Connector.open(url);
            connection.setRequestMethod(HttpConnection.GET);
            int code = connection.getResponseCode();
            if (code != HttpConnection.HTTP_OK) {
                throw new IOException("HTTP " + code);
            }
            input = connection.openInputStream();
            return readAll(input);
        } finally {
            if (input != null) {
                input.close();
            }
            if (connection != null) {
                connection.close();
            }
        }
    }

    private static String readAll(InputStream input) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[256];
        int read;
        while ((read = input.read(buffer)) != -1) {
            output.write(buffer, 0, read);
        }
        return new String(output.toByteArray());
    }
}
