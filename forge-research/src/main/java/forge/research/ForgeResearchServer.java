package forge.research;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import forge.GuiDesktop;
import forge.StaticData;
import forge.gui.GuiBase;
import forge.model.FModel;
import forge.research.service.ForgeRlServiceImpl;
import io.grpc.Server;
import io.grpc.ServerBuilder;

/**
 * Main entry point for the Forge Research gRPC server.
 * Initializes the Forge card database and starts a gRPC server
 * that accepts RL agent connections.
 *
 * Usage: ForgeResearchServer <port> [deck1.dck] [deck2.dck] ...
 *
 * When deck paths are provided, enables lazy card loading and only
 * loads the cards referenced by those decks. This reduces memory from
 * ~400MB to ~30-50MB per instance.
 */
public class ForgeResearchServer {

    private static final int DEFAULT_PORT = 50051;
    // Matches lines like "4 Lightning Bolt" or "4 Lightning Bolt|SET"
    private static final Pattern DECK_CARD_LINE = Pattern.compile("^\\d+\\s+(.+?)(?:\\|.*)?$");

    private Server server;

    public void start(int port) throws IOException {
        server = ServerBuilder.forPort(port)
                .addService(new ForgeRlServiceImpl())
                .build()
                .start();
        System.out.println("Forge Research Server started on port " + port);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutting down Forge Research Server...");
            ForgeResearchServer.this.stop();
        }));
    }

    public void stop() {
        if (server != null) {
            server.shutdown();
        }
    }

    public void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    public static void main(String[] args) throws Exception {
        int port = DEFAULT_PORT;
        List<String> deckPaths = new ArrayList<>();

        if (args.length > 0) {
            port = Integer.parseInt(args[0]);
        }
        for (int i = 1; i < args.length; i++) {
            deckPaths.add(args[i]);
        }

        boolean lazyLoad = !deckPaths.isEmpty();

        System.out.println("Initializing Forge" + (lazyLoad ? " (lazy card loading)..." : "..."));
        GuiBase.setInterface(new GuiDesktop());
        FModel.initialize(null, null, lazyLoad);

        if (lazyLoad) {
            preloadCardsFromDecks(deckPaths);
        }

        System.out.println("Card database loaded.");

        ForgeResearchServer server = new ForgeResearchServer();
        server.start(port);
        server.blockUntilShutdown();
    }

    /**
     * Parse deck files to extract card names, then load only those cards
     * into the StaticData card database via attemptToLoadCard().
     */
    private static void preloadCardsFromDecks(List<String> deckPaths) {
        Set<String> cardNames = new HashSet<>();

        for (String path : deckPaths) {
            try {
                List<String> lines = Files.readAllLines(new File(path).toPath());
                boolean inCardSection = false;
                for (String line : lines) {
                    String trimmed = line.trim();
                    if (trimmed.startsWith("[")) {
                        // Section header — card sections are [Main], [Sideboard], etc.
                        String section = trimmed.toLowerCase();
                        inCardSection = section.equals("[main]") || section.equals("[sideboard]");
                        continue;
                    }
                    if (inCardSection && !trimmed.isEmpty()) {
                        Matcher m = DECK_CARD_LINE.matcher(trimmed);
                        if (m.matches()) {
                            cardNames.add(m.group(1).trim());
                        }
                    }
                }
            } catch (IOException e) {
                System.err.println("Warning: could not read deck file: " + path + " — " + e.getMessage());
            }
        }

        System.out.println("Preloading " + cardNames.size() + " unique cards from " + deckPaths.size() + " decks...");
        StaticData db = FModel.getMagicDb();
        for (String name : cardNames) {
            db.attemptToLoadCard(name);
        }

        // Scan loaded card files for TokenScript$ references and preload those tokens
        String cardDir = forge.localinstance.properties.ForgeConstants.CARD_DATA_DIR;
        Set<String> tokenNames = new HashSet<>();
        Pattern tokenPattern = Pattern.compile("TokenScript\\$\\s*([\\w,]+)");
        for (String name : cardNames) {
            String fileName = name.toLowerCase().replace(" ", "_").replace("'", "").replace(",", "");
            String firstChar = fileName.substring(0, 1);
            File cardFile = new File(cardDir + firstChar + File.separator + fileName + ".txt");
            if (cardFile.exists()) {
                try {
                    for (String line : Files.readAllLines(cardFile.toPath())) {
                        Matcher m = tokenPattern.matcher(line);
                        if (m.find()) {
                            for (String token : m.group(1).split(",")) {
                                tokenNames.add(token.trim());
                            }
                        }
                    }
                } catch (IOException ignored) {}
            }
        }
        if (!tokenNames.isEmpty()) {
            System.out.println("Preloading " + tokenNames.size() + " tokens: " + tokenNames);
            for (String token : tokenNames) {
                db.attemptToLoadToken(token);
            }
        }

        System.out.println("Card preloading complete.");
    }
}
