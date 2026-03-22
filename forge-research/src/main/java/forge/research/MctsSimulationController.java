package forge.research;

import ai.onnxruntime.OrtException;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.player.Player;
import forge.research.onnx.MaskedSampler;
import forge.research.onnx.ObservationConverter;
import forge.research.onnx.OnnxInferenceEngine;
import forge.research.proto.DecisionPoint;
import forge.research.proto.Observation;

/**
 * PlayerController for MCTS simulation rollouts.
 * Like OnnxPlayerController but silent (no logging) and tracks agent decision count.
 * Used in cloned games where both players run ONNX inference locally.
 */
public class MctsSimulationController extends RlPlayerController {

    private final OnnxInferenceEngine engine;
    private final ObservationConverter converter = new ObservationConverter();
    private final MaskedSampler sampler = new MaskedSampler();
    private final boolean isAgent;
    private int agentDecisionCount = 0;
    private final int maxAgentDecisions;

    public MctsSimulationController(Game game, Player p, LobbyPlayer lp,
            int playerIndex, OnnxInferenceEngine engine,
            boolean isAgent, int maxAgentDecisions) {
        super(game, p, lp, playerIndex);
        this.engine = engine;
        this.isAgent = isAgent;
        this.maxAgentDecisions = maxAgentDecisions;
    }

    public int getAgentDecisionCount() {
        return agentDecisionCount;
    }

    @Override
    protected int queryAgent(DecisionPoint decisionPoint) {
        if (decisionPoint.getLegalActionsCount() == 1) {
            return 0;
        }

        // Track agent decisions and stop simulation when limit reached
        if (isAgent) {
            agentDecisionCount++;
            if (agentDecisionCount > maxAgentDecisions) {
                // Signal to stop by ending the game
                getGame().setGameOver(forge.game.GameEndReason.Draw);
                return 0;
            }
        }

        try {
            Game game = getGame();
            Player opponent = null;
            for (Player p : game.getPlayers()) {
                if (p != player) {
                    opponent = p;
                    break;
                }
            }
            if (opponent == null) {
                opponent = player;
            }

            Observation obs = observationBuilder.buildObservation(game, player, opponent);
            float[][] inputs = converter.allInputs(obs, decisionPoint);
            byte[] mask = converter.actionMask(decisionPoint);
            float[] logits = engine.infer(inputs);
            return sampler.sample(logits, mask);
        } catch (OrtException e) {
            return 0;
        }
    }
}
