import { describe, expect, it } from "vitest";
import { analyse, detectScript, isEnglish, scriptProfile } from "../src/lang/analyze.js";

describe("detectScript", () => {
  it("english → latin", () => {
    expect(detectScript("The invoice was paid twice and we need a refund.")).toBe("latin");
  });
  it("devanagari detected", () => {
    expect(detectScript("मुझसे दो बार शुल्क लिया गया")).toBe("devanagari");
  });
  it("cyrillic detected", () => {
    expect(detectScript("Счёт был оплачен дважды")).toBe("cyrillic");
  });
  it("han detected", () => {
    expect(detectScript("发票重复扣款")).toBe("han");
  });
  it("no letters → unknown", () => {
    expect(detectScript("12345 6789!")).toBe("unknown");
  });
});

describe("scriptProfile", () => {
  it("mixed text yields fractions", () => {
    const prof = scriptProfile("hello мир");
    expect(prof.latin).toBeGreaterThan(0);
    expect(prof.cyrillic).toBeGreaterThan(0);
    expect(prof.latin! + prof.cyrillic!).toBeCloseTo(1, 6);
  });
});

describe("analyse / isEnglish", () => {
  it("english text is english", () => {
    expect(isEnglish("We were billed twice for March, please refund the duplicate today.")).toBe(true);
  });
  it("hindi state is not english (dict state flattened)", () => {
    expect(isEnglish({ body: "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।" })).toBe(false);
  });
  it("german latin text routes away from english", () => {
    const det = analyse("Der Kunde wurde zweimal belastet und wir werden die Zahlung prüfen.");
    expect(det.script).toBe("latin");
    expect(det.language).toBe("de");
    expect(det.isEnglish).toBe(false);
  });
  it("nested JSON state: keys ignored, values used", () => {
    const det = analyse({ from: "user@acme.com", subject: "_facture 2024_", body: "bonjour, nous avons été facturés deux fois pour le mois de mars" });
    expect(det.script).toBe("latin");
    expect(det.isEnglish).toBe(false);
  });
  it("numeric-only state → unknown, treated as english-safe", () => {
    const det = analyse({ amount: "4411", id: 12345 });
    expect(det.script).toBe("unknown");
    expect(det.isEnglish).toBe(true);
  });
  it("short latin text stays undecided-but-english-safe (laya parity)", () => {
    expect(isEnglish("ok")).toBe(true);
  });
});
