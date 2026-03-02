export type Session = {
  chatId: string;
  cwd: string;
  model?: string;
  updatedAt: string;
};

export class SessionStore {
  private readonly sessions = new Map<string, Session>();

  create(sessionId: string, session: Omit<Session, "updatedAt">): Session {
    const entry: Session = {
      ...session,
      updatedAt: new Date().toISOString(),
    };
    this.sessions.set(sessionId, entry);
    return entry;
  }

  get(sessionId: string): Session | undefined {
    return this.sessions.get(sessionId);
  }

  has(sessionId: string): boolean {
    return this.sessions.has(sessionId);
  }

  update(sessionId: string, patch: Partial<Omit<Session, "chatId" | "cwd">>): Session {
    const existing = this.sessions.get(sessionId);
    if (!existing) {
      throw new Error(`Unknown session: ${sessionId}`);
    }
    const next: Session = {
      ...existing,
      ...patch,
      updatedAt: new Date().toISOString(),
    };
    this.sessions.set(sessionId, next);
    return next;
  }

  list(): Array<{ sessionId: string; session: Session }> {
    return Array.from(this.sessions.entries()).map(([sessionId, session]) => ({
      sessionId,
      session,
    }));
  }
}
