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

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * PlayerController that runs ONNX inference locally instead of blocking on gRPC queues.
 * Extends RlPlayerController — all 900+ lines of decision-building logic are inherited.
 * Automatically logs every decision to a JSON file in plays/.
 */
public class OnnxPlayerController extends RlPlayerController {

    private final OnnxInferenceEngine engine;
    private final ObservationConverter converter = new ObservationConverter();
    private final MaskedSampler sampler = new MaskedSampler();
    private PrintWriter logWriter;
    private boolean firstEntry = true;

    public OnnxPlayerController(Game game, Player p, LobbyPlayer lp,
            int playerIndex, OnnxInferenceEngine engine) {
        super(game, p, lp, playerIndex);
        this.engine = engine;
        initLog();
    }

    private void initLog() {
        try {
            File dir = new File("plays");
            dir.mkdirs();
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            File logFile = new File(dir, "onnx_game_" + timestamp + ".jsonl");
            logWriter = new PrintWriter(new FileWriter(logFile), true);
            System.out.println("Logging ONNX decisions to: " + logFile.getAbsolutePath());
        } catch (IOException e) {
            System.err.println("Failed to create ONNX log file: " + e.getMessage());
        }
    }

    @Override
    protected int queryAgent(DecisionPoint decisionPoint) {
        // Auto-resolve single legal action (inherited check)
        if (decisionPoint.getLegalActionsCount() == 1) {
            logDecision(decisionPoint, 0, true);
            return 0;
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

            // Convert to float arrays for ONNX
            float[][] inputs = converter.allInputs(obs, decisionPoint);
            byte[] mask = converter.actionMask(decisionPoint);

            // Run ONNX inference
            float[] logits = engine.infer(inputs);

            // Sample action with mask
            int action = sampler.sample(logits, mask);

            logDecision(decisionPoint, action, false);
            return action;
        } catch (OrtException e) {
            System.err.println("ONNX inference failed: " + e.getMessage());
            return 0; // fallback: first action
        }
    }

    private void logDecision(DecisionPoint dp, int action, boolean autoResolved) {
        if (logWriter == null) return;
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"turn\":").append(getGame().getPhaseHandler().getTurn());
            sb.append(",\"phase\":\"").append(getGame().getPhaseHandler().getPhase()).append("\"");
            sb.append(",\"type\":\"").append(dp.getType()).append("\"");
            sb.append(",\"prompt\":\"").append(escapeJson(dp.getPrompt())).append("\"");
            sb.append(",\"num_actions\":").append(dp.getLegalActionsCount());
            sb.append(",\"chosen\":").append(action);
            String chosenDesc = action < dp.getLegalActionsCount()
                    ? dp.getLegalActions(action).getDescription() : "???";
            sb.append(",\"chosen_desc\":\"").append(escapeJson(chosenDesc)).append("\"");
            sb.append(",\"auto_resolved\":").append(autoResolved);
            sb.append(",\"agent_life\":").append(player.getLife());
            Player opp = null;
            for (Player p : getGame().getPlayers()) {
                if (p != player) { opp = p; break; }
            }
            sb.append(",\"opp_life\":").append(opp != null ? opp.getLife() : -1);

            // Log all legal actions
            sb.append(",\"actions\":[");
            for (int i = 0; i < dp.getLegalActionsCount(); i++) {
                if (i > 0) sb.append(",");
                sb.append("\"").append(escapeJson(dp.getLegalActions(i).getDescription())).append("\"");
            }
            sb.append("]");

            sb.append("}");
            logWriter.println(sb.toString());
        } catch (Exception e) {
            // Don't let logging errors break gameplay
        }
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r");
    }
}
