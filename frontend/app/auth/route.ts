import { NextRequest, NextResponse } from "next/server";
export async function POST(request: NextRequest) {
  const { token } = await request.json();
  if (typeof token !== "string" || token.length > 512) return new Response("Invalid token", {status:400});
  const checked = await fetch(`${process.env.API_PROXY_TARGET || "http://localhost:8088"}/auth/whoami`, {headers:{Authorization:`Bearer ${token}`},cache:"no-store"});
  if (!checked.ok) return new Response("Sign-in failed", {status:401});
  const result=NextResponse.json(await checked.json());
  result.cookies.set("session-token",token,{httpOnly:true,sameSite:"strict",secure:request.nextUrl.protocol==="https:",path:"/",maxAge:3600});
  return result;
}
export async function DELETE() {
  const result=NextResponse.json({status:"signed out"});result.cookies.delete("session-token");return result;
}
