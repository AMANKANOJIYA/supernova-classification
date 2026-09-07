"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import useSWR from "swr";
import {
  LayoutDashboard,
  Radio,
  Orbit,
  BarChart3,
  Sparkles,
  BookOpen,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
} from "@/components/ui/sidebar";
import { api } from "@/lib/api";
import { fmtCompact } from "@/lib/format";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/live", label: "Live monitor", icon: Radio },
  { href: "/objects", label: "Objects", icon: Orbit },
  { href: "/results", label: "Results", icon: BarChart3 },
  { href: "/anomalies", label: "Anomalies", icon: Sparkles },
  { href: "/about", label: "About", icon: BookOpen },
];

export function AppSidebar() {
  const pathname = usePathname();
  const { data: health } = useSWR("health", () => api.health(), {
    refreshInterval: 15000,
    shouldRetryOnError: false,
  });
  const { data: stats } = useSWR("stats-sidebar", () => api.stats(), {
    refreshInterval: 15000,
    shouldRetryOnError: false,
  });

  const streamActive = health?.status === "ok";

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="px-3 py-4">
        <div className="flex items-center gap-2.5 px-1">
          <span className="text-xl leading-none" aria-hidden>
            🔭
          </span>
          <span className="text-card-title font-semibold tracking-tight group-data-[collapsible=icon]:hidden">
            SN Classifier
          </span>
        </div>
      </SidebarHeader>
      <SidebarSeparator />
      <SidebarContent className="px-2 py-3">
        <SidebarMenu>
          {NAV.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <SidebarMenuItem key={item.href}>
                <SidebarMenuButton asChild isActive={active} tooltip={item.label} className="h-10 text-base">
                  <Link href={item.href}>
                    <item.icon className="size-4.5" />
                    <span>{item.label}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            );
          })}
        </SidebarMenu>
      </SidebarContent>
      <SidebarSeparator />
      <SidebarFooter className="gap-2 px-3 py-4 group-data-[collapsible=icon]:items-center">
        <div className="flex items-center gap-2 text-caption">
          <span
            className={`inline-block size-2 rounded-full ${
              streamActive ? "bg-ii shadow-[0_0_0_3px_var(--ii)]/20 animate-pulse" : "bg-muted-foreground/40"
            }`}
            style={streamActive ? { boxShadow: "0 0 0 3px color-mix(in oklch, var(--ii) 25%, transparent)" } : undefined}
          />
          <span className="group-data-[collapsible=icon]:hidden">
            stream: {streamActive ? "active" : "offline"}
          </span>
        </div>
        <div className="text-caption group-data-[collapsible=icon]:hidden">
          <span className="tabular-nums-all font-medium text-foreground">
            {stats ? fmtCompact(stats.n_objects) : "—"}
          </span>{" "}
          objects
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
