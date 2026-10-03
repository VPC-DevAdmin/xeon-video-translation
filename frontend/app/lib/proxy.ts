import { NextRequest } from "next/server";

export async function proxy(request: NextRequest, parts: string[], target: string, internalKey?: string) {
  // Path segments are framework-decoded. Reject traversal and encoded separators.
  if (parts.some(p => p === "." || p === ".." || /[\\/]/.test(p))) {
    return new Response("Invalid path", { status: 400 });
  }
  const headers = new Headers();
  for (const name of ["content-type", "last-event-id", "range"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const token=request.cookies.get("session-token")?.value;
  if (token) headers.set("Authorization",`Bearer ${token}`);
  try {
  if (internalKey) {
    const identity=await fetch(`${process.env.API_PROXY_TARGET || "http://localhost:8088"}/auth/whoami`,{headers:token?{Authorization:`Bearer ${token}`}:{},cache:"no-store"});
    if (!identity.ok) return new Response("Authentication required",{status:401});
    headers.set("x-owner-id",(await identity.json()).owner);
    headers.set("x-internal-key", internalKey);
  }
    const response = await fetch(`${target}/${parts.map(encodeURIComponent).join("/")}${request.nextUrl.search}`, {
      method: request.method, headers, cache: "no-store", redirect: "manual",
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      signal: request.signal, duplex: "half",
    } as RequestInit);
    const outgoing = new Headers();
    for (const name of ["content-type", "content-length", "content-range", "accept-ranges", "content-disposition"]) {
      const value = response.headers.get(name);
      if (value) outgoing.set(name, value);
    }
    outgoing.set("Cache-Control", "no-store");
    outgoing.set("X-Accel-Buffering", "no");
    return new Response(response.body, { status: response.status, headers: outgoing });
  } catch {
    return new Response("Upstream unavailable", { status: 502 });
  }
}
