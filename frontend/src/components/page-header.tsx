export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="flex flex-col gap-2">
        <h1 className="text-page-title">{title}</h1>
        {description ? <p className="text-body max-w-2xl text-muted-foreground">{description}</p> : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function SectionHeading({
  title,
  description,
  actions,
  className,
}: {
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`mb-5 flex flex-wrap items-end justify-between gap-3 ${className ?? ""}`}>
      <div className="flex flex-col gap-1">
        <h2 className="text-section-heading">{title}</h2>
        {description ? <p className="text-caption max-w-2xl">{description}</p> : null}
      </div>
      {actions}
    </div>
  );
}
