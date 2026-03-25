package forge.research.observation;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Maps card names to stable integer IDs for observation encoding.
 * Thread-safe. IDs are deterministic — derived from a hash of the card name,
 * so they are consistent across JVM restarts and server processes.
 *
 * IDs are in range [1, VOCAB_SIZE-1]. ID 0 is reserved for padding.
 */
public class CardRegistry {

    private static final CardRegistry INSTANCE = new CardRegistry();
    private static final int VOCAB_SIZE = 4096;

    private final ConcurrentHashMap<String, Integer> nameToId = new ConcurrentHashMap<>();

    public static CardRegistry getInstance() {
        return INSTANCE;
    }

    public int getNameId(String cardName) {
        return nameToId.computeIfAbsent(cardName, k -> {
            // Deterministic hash mapped to [1, VOCAB_SIZE-1]
            int hash = k.hashCode() & 0x7FFFFFFF; // ensure positive
            int id = 1 + (hash % (VOCAB_SIZE - 1));
            // Check for hash collisions
            for (var entry : nameToId.entrySet()) {
                if (entry.getValue() == id && !entry.getKey().equals(k)) {
                    System.err.println("[CardRegistry] COLLISION: '" + k + "' and '" +
                        entry.getKey() + "' both map to ID " + id +
                        " (vocab size " + VOCAB_SIZE + ", " + nameToId.size() + " cards registered)");
                }
            }
            return id;
        });
    }

    public int size() {
        return nameToId.size();
    }
}
