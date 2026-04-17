/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    // Allow large multipart uploads through the dev server proxy.
    serverActions: { bodySizeLimit: "200mb" },
  },
};

module.exports = nextConfig;
