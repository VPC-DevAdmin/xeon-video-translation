import { NextRequest } from "next/server";
import { proxy } from "../../lib/proxy";
export const dynamic = "force-dynamic";
const handler = async (r: NextRequest, c: { params: Promise<{ path: string[] }> }) =>
  proxy(r, (await c.params).path, process.env.INGEST_PROXY_TARGET || "http://localhost:8091", process.env.INTERNAL_API_KEY);
export { handler as GET, handler as POST, handler as DELETE };
