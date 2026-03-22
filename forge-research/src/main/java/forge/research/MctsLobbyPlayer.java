package forge.research;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.player.IGameEntitiesFactory;
import forge.game.player.Player;
import forge.game.player.PlayerController;
import forge.research.onnx.OnnxInferenceEngine;

/**
 * LobbyPlayer for MCTS simulation games.
 * Creates MctsSimulationController instances that run ONNX inference
 * and track agent decision counts for simulation depth control.
 */
public class MctsLobbyPlayer extends LobbyPlayer implements IGameEntitiesFactory {

    private final OnnxInferenceEngine engine;
    private final int playerIndex;
    private final boolean isAgent;
    private final int maxAgentDecisions;

    public MctsLobbyPlayer(String name, OnnxInferenceEngine engine,
            int playerIndex, boolean isAgent, int maxAgentDecisions) {
        super(name);
        this.engine = engine;
        this.playerIndex = playerIndex;
        this.isAgent = isAgent;
        this.maxAgentDecisions = maxAgentDecisions;
    }

    @Override
    public Player createIngamePlayer(Game game, int id) {
        Player p = new Player(getName(), game, id);
        p.setFirstController(new MctsSimulationController(
                game, p, this, playerIndex, engine, isAgent, maxAgentDecisions));
        return p;
    }

    @Override
    public PlayerController createMindSlaveController(Player master, Player slave) {
        return new MctsSimulationController(
                slave.getGame(), slave, this, playerIndex, engine, isAgent, maxAgentDecisions);
    }

    @Override
    public void hear(LobbyPlayer player, String message) {
        // Simulation doesn't need chat
    }
}
