You are compressing a 65,887-character system prompt for a 77-class banking
intent classifier. The accuracy of the classifier on a held-out 200-sample
slice of PolyAI/banking77 is constrained by the eval — if accuracy drops below
the baseline minus 2 percentage points, the metric is penalized by
+10,000,000 and the variant is discarded.

Your goal is to **minimize the character count** of the SYSTEM_PROMPT string
in optimize.py while keeping accuracy above the threshold.

Levers to try, ordered roughly from cheapest-wins to most-impactful:

1. **Strip the verbose preamble.** The Operating Principles, Output Format
   Specification, and Disambiguation Guidelines sections are heavily
   redundant with the per-class blocks and with each other. Consolidate
   into a single concise rule list.

2. **Strip per-class fluff.** Each class block has a verbose Description,
   Typical phrasings, Disambiguation, and Output requirement. The Output
   requirement is identical across all 77 classes. Most Descriptions can
   be reduced to the topic phrase. Most Typical phrasings can be cut
   entirely — the LLM doesn't need example user utterances when the topic
   is well-named.

3. **Drop the FAQ and Common Mistakes sections.** They restate rules from
   the preamble.

4. **Drop or shorten the Worked Examples.** They demonstrate format
   compliance, which the LLM already learns from the label list.

5. **Compress the label list.** A bare enumeration of the 77 canonical
   labels may be enough — the model can disambiguate from label names
   alone for many of the 77 classes.

6. **Reformat for density.** Trade markdown structure for compact list or
   table format if it preserves accuracy.

What you MUST preserve:

- The exact 77 canonical label strings (case-sensitive, including
  `Refund_not_showing_up` with capital R and `reverted_card_payment?` with
  trailing question mark).
- An instruction telling the model to output exactly one canonical label
  string with no commentary.
- The classify() function signature, the OpenAI call, and the surrounding
  Python code outside the WECO-MUTABLE REGION markers.

What you MUST NOT do:

- Change the variable name `SYSTEM_PROMPT` or its assignment.
- Modify any code outside the WECO-MUTABLE REGION markers in optimize.py.
- Drop labels from the taxonomy (the eval will fail with parse errors).

Iterate aggressively. Real production prompts of this shape compress
80-95% with no accuracy loss; large jumps in a single mutation are
expected and welcomed.
