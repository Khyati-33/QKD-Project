# Figures mapped to the routing-paper structure

This plan follows the figure sequence and broad section placement in Johann et al. The diagrams will be redrawn for this project; figures from the source paper will not be copied.

| Routing-paper figure role | QKD project figure | Source and status |
|---|---|---|
| Fig. 1, link key rate versus fiber distance | Modeled link-rate/QBER proxy versus distance for fiber and FSO | Generate from `physics.py`; label as scenario output. Do not overlay the source paper's curve as validation because device and protocol assumptions differ. |
| Fig. 2, trusted-node key relaying | TN and STN chain schematic with end-user post-processing | New conceptual diagram. Explicitly state the STN finite-key proof is cited background and is not implemented by the simulator. |
| Fig. 3, network architecture | Application, controller, key-management, QKD link, and physical-channel layers | New conceptual system diagram, with implemented and proposed components visually distinguished. |
| Fig. 4, network topology | Seven city hubs, road-corridor proxies, fiber links, candidate FSO links, and example demands | Generate from `topology.py`; current graph is 2,370 nodes and 2,653 links. The geometry is not surveyed infrastructure. |
| Fig. 5, simulation flow | Scenario/configuration, environment reset, policy or baseline, constraints, transition, metrics | New flowchart matching the actual simulation pipeline. |
| Fig. 6, LSTM prediction diagnostic | LSTM-PPO versus GNN-PPO policy or training diagnostic | Await matched LSTM training. Current LSTM encoder reads an ordered node-feature array, not temporal demand history. |
| Fig. 7, blocked-demand comparison | Route failure/success or blocking probability across BFS, Dijkstra, Max-SKR, GNN, LSTM, weight optimizer, and ILP | Await common demand and resource model. Current ten-seed single-request comparison is too narrow for the reference paper's blocking metric. |
| Fig. 8, example key-store trajectory | Per-link normalized pool state during an episode | Requires instrumented runs. Current pool is a normalized simulator state, not a calibrated key count. |
| Fig. 9, mean key-store utilization | Aggregate pool level by edge/corridor and method | Requires multi-demand simulator and matched policy runs. |
| Fig. 10, management traffic | Controller queries or telemetry volume by method | Not currently modeled. Add a message-count model before reporting quantitative results. |

Current reproducible assets include a corridor map, architecture sketch, and a compact link-model summary. The final paper should prefer the first five figures as system and method figures, then add result figures only when their corresponding comparison is complete. Unsupported result slots belong in the draft plan or appendix, not as fabricated plots.
