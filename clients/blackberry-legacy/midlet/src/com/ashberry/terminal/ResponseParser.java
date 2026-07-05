package com.ashberry.terminal;

public final class ResponseParser {
    private ResponseParser() {
    }

    public static String compact(String response) {
        if (response == null || response.length() == 0) {
            return "EMPTY RESPONSE";
        }
        if (response.startsWith("OK\n")) {
            return response.substring(3);
        }
        return response;
    }
}
