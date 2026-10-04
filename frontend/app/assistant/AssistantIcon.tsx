type IconName = "transcript" | "person" | "settings" | "microphone" | "camera" | "close" | "stop" | "interrupt";

export default function AssistantIcon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    transcript: <><path d="M4 5h16M4 10h16M4 15h12M4 20h9" /></>,
    person: <><circle cx="10" cy="8" r="3.5" /><path d="M3 20c0-3.7 3.1-6 7-6 2.2 0 4.2.8 5.5 2.2M19 13v7M15.5 16.5h7" /></>,
    settings: <><circle cx="12" cy="12" r="3" /><path d="M19 13.5v-3l-2-.7-.6-1.4.9-1.9-2.1-2.1-1.9.9-1.4-.6L11.2 3h-3l-.7 2-1.4.6-1.9-.9-2.1 2.1.9 1.9-.6 1.4-2 .7v3l2 .7.6 1.4-.9 1.9 2.1 2.1 1.9-.9 1.4.6.7 2h3l.7-2 1.4-.6 1.9.9 2.1-2.1-.9-1.9.6-1.4z" /></>,
    microphone: <><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8" /></>,
    camera: <><rect x="3" y="5" width="14" height="14" rx="2" /><path d="m17 9 4-2v10l-4-2" /></>,
    close: <path d="M5 5 19 19M19 5 5 19" />,
    stop: <rect x="7" y="7" width="10" height="10" rx="1" />,
    interrupt: <><path d="M7 5v14M12 8v8M17 5v14" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}
