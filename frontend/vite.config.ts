import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",   // without this, Vite defaults to binding "localhost" only, which on
    // this machine resolved to the IPv6 loopback (::1) exclusively -- a browser pointed at
    // 127.0.0.1:5174 (or any client whose DNS/OS preferred IPv4 for "localhost") got a flat
    // connection-refused, which is what a "blank screen" actually was here (confirmed via
    // curl: 127.0.0.1:5174 refused, [::1]:5174 answered 200 OK). 0.0.0.0 binds every local
    // interface, IPv4 and IPv6 both.
    port: 5174,
    strictPort: true
  }
});

