# Contributing to Cleanhatch

Thanks for helping improve Cleanhatch. This guide covers bug reports, Tier 1
field-pitfall scenarios, and Tier 2 code contributions. It also states the
licensing terms in plain language.

## Short version

- You do **not** need to buy L1, own a license key, or accept the L1 commercial
  terms to report a bug, submit a Tier 1 scenario, or open a Tier 2 code PR.
- A bug report itself needs no copyright or patent license, no CLA, and no DCO.
  We ask only that you desensitize the report and let us discuss it publicly.
- Tier 1 scenario text and synthetic examples are contributed under
  [Apache-2.0](LICENSE), the same license as the L0 kernel. You keep your
  copyright; you grant the public the permissions in that license.
- Tier 2 code is contributed under [Apache-2.0](LICENSE) for files covered by
  the root license. No CLA is required.
- Apache-2.0 already permits commercial use. Using an Apache-2.0 contribution
  inside L1 requires compliance with Apache-2.0 (the `LICENSE` file, any
  `NOTICE` file, attribution, and change notices), not an extra CLA.
- We do not ask for a standing CLA or a blanket moral-rights waiver. If we ever
  need rights beyond Apache-2.0 for a specific contribution, we will propose a
  separate, specific, recorded agreement that you may accept or decline.
- Declining extra rights does not affect acceptance, attribution, or reward
  eligibility for an Apache-2.0 contribution.

The L0 kernel and other files covered by the root [LICENSE](LICENSE) are open
source under Apache-2.0. The L1 Pro Suite is distributed under separate
commercial terms. Contributing does not give you L1 rights, and buying L1 does
not change the license of your contribution.

## Before you start

- Search existing issues and pull requests to avoid duplicate work.
- Open an issue before a large refactor, a new public API, or a change that
  crosses the L0/L1 boundary.
- Keep each contribution focused. Unrelated cleanup makes review slower.
- Do not include proprietary data, credentials, client names, personal
  filesystem paths, or production exports. Use small synthetic examples.

## Bug reports

Use the **Bug report** issue form. A useful report contains:

- a short description of the defect;
- the Cleanhatch version, Python version, operating system, and install method;
- the smallest reproduction you can share, using synthetic or desensitized data;
- the expected result and the actual result, including the full sanitized
  traceback or verdict.

A bug report is not a request for a broad copyright or patent license. The form
asks you to confirm that you removed sensitive material and that the report may
be discussed publicly. It does not require a CLA, a DCO sign-off, or an L1
purchase.

If you include a patch or a code snippet in a bug report, that code is a
contribution and is licensed under Apache-2.0. The report text itself is not a
contribution to the codebase. If you do not want a snippet treated as a
contribution, mark it "Not a Contribution" in the report; we can still discuss
the defect.

Security-sensitive reports must not be filed publicly. Use the private security
advisory link in the issue chooser.

## Tier 1: Field Pitfall Scout (scenario)

Use the **Silent Killer / Field Pitfall** issue form when you have a real
production failure that stayed green. Tier 1 is a scenario contribution: you
describe the failure, the temporal coordinates, and a minimal synthetic poison
pattern; maintainers write the rule.

A good Tier 1 report includes:

- the symptom, including what the pipeline reported and what was actually wrong;
- generic schema shapes and the temporal role of each field (`event_time`,
  `knowledge_time`, `decision_time`, version or revision keys);
- a minimal reproduction or poison pattern with the wrong output and the
  expected guard verdict;
- the downstream impact and any detection gap, if useful;
- a desensitized attribution choice.

### Tier 1 license

When you check the license acceptance item in the form, you confirm that the
scenario text and any synthetic examples in that issue are licensed under
[Apache-2.0](LICENSE). The form shows this as an explicit acceptance item; the
license covers the issue content, not a future code pull request. You do not
transfer or assign your copyright.

If you cannot license the scenario under Apache-2.0, do not submit it through
the form. Contact the maintainers first to discuss whether a separate
arrangement is possible.

Tier 1 does not require code, a DCO sign-off, an L1 purchase, or any extra
rights. Accepted scenarios are eligible for the attribution and reward described
in the [README](README_PUBLIC.md).

## Tier 2: Core Recipe Architect (code PR)

Tier 2 is a code contribution: a vectorized rule, a canary poisoning test, or
another focused change to files covered by the root Apache-2.0 license. Open a
pull request using the [PR template](.github/PULL_REQUEST_TEMPLATE.md).

A good Tier 2 PR includes:

- a focused change with tests for the new behavior;
- a canary test that fails on the poisoned input and passes on the clean input;
- documentation for any public API or verdict change;
- the commands you ran and their results;
- no proprietary fixtures, secrets, or generated build artifacts.

### Tier 2 license

Code contributions to files covered by the root [LICENSE](LICENSE) are licensed
under Apache-2.0. No CLA is required. If a file carries a separate license or is
part of the commercial L1 distribution, the pull request will identify that
before review; do not submit it under Apache-2.0 terms you do not intend.

This repository does not currently require a Developer Certificate of Origin
(DCO) sign-off. A `Signed-off-by` line is welcome as a provenance record, but a
DCO sign-off is **not** a CLA, does not grant rights beyond Apache-2.0, and is
not a merge gate today. If a DCO policy is adopted, it will be documented here
and applied prospectively.

Tier 2 does not require an L1 purchase, a CLA, or any extra rights. Accepted
contributions are eligible for the attribution and reward described in the
[README](README_PUBLIC.md), subject to the program terms and the
`Contributor Edition` restrictions stated there.

## If we need rights beyond Apache-2.0

We do not maintain a standing CLA and we do not ask for a blanket waiver of
moral rights. Apache-2.0 is sufficient for the normal open-source and commercial
use of a contribution, including use inside L1, when the license, `NOTICE`,
attribution, and change-notice obligations are met.

If a concrete need for broader rights arises for a specific contribution, we
will use this process:

1. A maintainer sends a separate written proposal that identifies the exact
   contribution (issue, PR, commit, or file), the exact additional rights
   requested, the exact agreement text, its version, and its scope and term.
2. You accept that exact text and version through a recorded affirmative action,
   such as a comment on the issue or PR that names the agreement and version, or
   a separately signed document.
3. Only the identified contribution and the identified rights are affected.
   Acceptance does not cover your other contributions or future work.
4. We keep the proposal and your acceptance record with the contribution.
5. If you decline, we will not use that contribution under the proposed broader
   terms. Declining is not by itself a reason to reject the contribution: we
   will still review it, attribute it, and consider it for rewards under
   Apache-2.0. Mergeability is decided separately on objective project reasons
   such as correctness, scope, tests, or data hygiene; those technical review
   requirements are not conditions for extra rights.

### What is not consent

None of the following grants rights beyond Apache-2.0:

- buying L1, accepting a checkout, or accepting the L1 commercial terms;
- a generic "I agree" checkbox that does not display the exact agreement text
  and version;
- a statement that you are willing to be contacted;
- silence, a merged pull request, or a maintainer's request;
- a DRAFT purchase document's "by submitting" wording. The L1 terms govern L1
  purchases, not contributions to this repository.

A checkbox is an acceptance only when the exact additional-rights text and
version are shown next to it. A request for extra rights is never a condition of
normal Apache-2.0 contribution, review, attribution, or reward eligibility.

## Attribution and anonymity

Credit is available as:

- a public GitHub `@handle`;
- a custom pseudonym such as `QuantAnon-42`;
- `Anonymous`, written as text in the attribution field.

`Anonymous` means we will not publish a name or handle in the attribution we
control. We cannot make your GitHub account anonymous: GitHub records the author
of an issue or pull request, and public Git history may contain commit author
metadata. If that matters, use a pseudonymous GitHub account, or contribute the
scenario through the issue form without attaching code.

We apply credit manually in the documentation, release notes, and the
`/silent-killers/` registry where applicable. This guide does not add a
self-serve registry or a website account feature.

## Data hygiene

- Replace client, fund, employer, and vendor names with generic labels.
- Replace real identifiers with synthetic ones that preserve the shape of the
  failure.
- Remove credentials, tokens, connection strings, account numbers, personal
  names, email addresses, and local filesystem paths.
- Reduce examples to the smallest table that still shows the failure.
- If you accidentally include sensitive material, say so in the issue without
  repeating it. Maintainers may edit or remove the content.

## Tests and checks

Run the project checks before opening a pull request:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Add a regression test for a bug fix and a behavior test for a new rule. Tier 2
rule contributions also need a canary test that proves the poisoned input is
caught and the clean input passes. Tests must not call the network or require a
license key.

## Review, acceptance, and rewards

Maintainers review contributions for correctness, scope, tests, and data
hygiene. We may ask for changes, edit for clarity, or decline a contribution
that does not meet those objective project requirements. That general review
right is independent of any extra-rights decision. Merging a contribution means
it is accepted under Apache-2.0; it does not create a support contract or a
delivery commitment.

Tier 1 and Tier 2 rewards are described in the [README](README_PUBLIC.md). They
are subject to the stated program terms, caps, and the `Contributor Edition`
restrictions. No purchase and no extra-rights acceptance is required to be
eligible. Rewards have no cash value and are not compensation for a copyright
assignment.

Apache-2.0 rights are irrevocable. If a credit is wrong, tell us and we will
correct the attribution where practicable.

## Links

- [Apache-2.0 license](LICENSE)
- [README and contribution tiers](README_PUBLIC.md)
- [Bug report form](.github/ISSUE_TEMPLATE/bug_report.yml)
- [Silent Killer / Field Pitfall form](.github/ISSUE_TEMPLATE/silent_killer_pitfall.yml)
- [Pull request template](.github/PULL_REQUEST_TEMPLATE.md)
- [Private security advisory](https://github.com/cleanhatch/cleanhatch/security/advisories/new)

This guide may be updated. Material changes apply prospectively; an
Apache-2.0 grant that already applies to a contribution is not withdrawn.
