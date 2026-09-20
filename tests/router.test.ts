import { describe, expect, it } from "vitest";
import { matchTypedDecisionsWorkflow, normaliseName, route } from "../src/router/route.js";
import type { Questions } from "../src/core/types.js";

const genericQuestions: Questions = {
  department: {
    type: "choice",
    instructions: "Which department?",
    criteria: { billing: "invoices", tech: "bugs" },
  },
  urgent: { type: "noul", instructions: "Is it urgent?" },
};

describe("normaliseName", () => {
  it("resolves aliases", () => {
    expect(normaliseName("en")).toBe("english");
    expect(normaliseName("multi")).toBe("multilingual");
    expect(normaliseName("typed")).toBe("typed-decisions");
    expect(() => normaliseName("nope")).toThrow(/unknown model/);
  });
});

describe("route precedence", () => {
  it("explicit model wins", () => {
    const d = route({ body: "मुझसे दो बार शुल्क" }, genericQuestions, { model: "english" });
    expect(d.model).toBe("english");
    expect(d.reason).toContain("explicit model");
  });

  it("explicit task wins over detection", () => {
    const d = route("english text", genericQuestions, { task: "typed_decisions" });
    expect(d.model).toBe("typed-decisions");
  });

  it("workflow match only with opt-in", () => {
    const cs: Questions = {
      action: { type: "choice", instructions: "a", criteria: ["x", "y"] },
      category: { type: "choice", instructions: "c", criteria: ["p", "q"] },
      churn_risk: { type: "noul", instructions: "r" },
      needs_human: { type: "noul", instructions: "h" },
      urgency: { type: "score", instructions: "u", criteria: ["0", "1", "2"] },
    };
    expect(matchTypedDecisionsWorkflow(cs)).toBe("customer_service");
    expect(route("hello", cs).model).toBe("english");
    expect(route("hello", cs, { autoTaskDetection: true }).model).toBe("typed-decisions");
  });

  it("explicit lang beats script detection", () => {
    const d = route({ body: "मुझसे दो बार शुल्क लिया गया" }, genericQuestions, { lang: "en" });
    expect(d.model).toBe("english");
    expect(d.reason).toContain("explicit lang");
  });

  it("non-latin script → multilingual with reason", () => {
    const d = route({ body: "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।" }, genericQuestions);
    expect(d.model).toBe("multilingual");
    expect(d.reason).toContain("non-Latin script (devanagari");
    expect(d.repo).toBe("convaiinnovations/laya/multilingual");
  });

  it("latin non-english → multilingual", () => {
    const d = route("Der Kunde wurde zweimal belastet und wir werden die Zahlung nicht prüfen.", genericQuestions);
    expect(d.model).toBe("multilingual");
    expect(d.reason).toContain("language looks like 'de'");
  });

  it("english → english checkpoint", () => {
    const d = route("We were billed twice for March, please refund the duplicate today.", genericQuestions);
    expect(d.model).toBe("english");
    expect(d.reason).toBe("English Latin text");
  });

  it("no letters → default", () => {
    const d = route({ amount: "4411" }, genericQuestions);
    expect(d.model).toBe("english");
    expect(d.reason).toContain("no letters detected");
  });
});
