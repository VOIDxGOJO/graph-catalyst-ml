import "./globals.css";

export const metadata = {
  title: "Catalyst Designer",
  description: "Graph-based ML demo — catalyst recommendation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-b from-white to-slate-50">
        {children}
      </body>
    </html>
  );
}
