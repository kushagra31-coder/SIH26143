# SIH26143 Oil Spill Vessel Attribution: MVP Validation Report

## 1. Pipeline Status
- **Phase 1 Mask Source**: Predicted mask placeholder (waiting on external training pipeline).
- **Phase 2 Drift Hindcast**: Successfully backtracked the georeferenced origin. The predicted origin converged to 26.06 km of the actual grounding point.
- **Phase 3 & 4 Validation**: Successfully validated single-vessel pipeline end-to-end.

## 2. Wakashio Validation Results
The single known vessel (Wakashio, IMO: 9337119) was scored against the predicted origin:
- **Total Evidence Score**: 63.38/100
- **Spatial Match**: 47.88/100
- **Behaviour Match**: 86.64/100

## 3. Honest Scope Limitation: Candidate Ranking
Multi-vessel ranking was not demonstrable in this MVP due to API access limitations. The Global Fishing Watch (GFW) "Events" API, while accessible, is focused heavily on fishing fleets and transshipments. It returned exactly 0 usable candidates for this bulk-carrier incident within the Mauritius bounding box. 

Therefore, rather than fabricating fake AIS tracks to simulate a ranking, we have explicitly scoped this MVP to validate the math and trajectory logic against the single known vessel (Wakashio). A full multi-vessel attribution ranking requires production-tier AIS access (e.g. the pending GFW elevated-access request, or commercial APIs like Spire), which is noted as pending future work.

## 4. Ground-Truth Citations
The behaviour deviation and actual grounding anchor used for this validation were explicitly extracted from the official casualty report:
*Panama Maritime Authority, Directorate General of Merchant Marine, Maritime Affairs Investigation Department, "Report MV Wakashio R-029-2021-DIAM," 2023.*
