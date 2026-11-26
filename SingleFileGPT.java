import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.DoubleSupplier;

/**
 * SingleFileGPT: a decoder-only transformer language model in pure Java (one file, zero dependencies).
 *
 *  - tokenizer: byte-level (vocab 256) or GPT-2 regex byte-level BPE (vocab > 256)
 *  - architecture: LayerNorm | RMSNorm, GELU | SwiGLU, learned positions | RoPE
 *  - hand-derived backprop for every op, verified by central finite differences in selftest
 *  - AdamW (bit-exact checkpoint resume), gradient accumulation, residual dropout, disk streaming
 *  - weight-only int8 & BitNet-style ternary QAT, byte-resident inference, row-scale kernel, spin-wait threads
 *  - KV-cached generation with RoPE cache trimming, sliding-window held-out evaluation, thread-sweep bench
 *
 *   java SingleFileGPT.java selftest
 *   java SingleFileGPT.java bench dim=512 heads=8 layers=8 ctx=128 len=60
 *   java SingleFileGPT.java train data=corpus.txt out=model.bin vocab=512 dim=128 heads=4 layers=4 ctx=128
 *   java SingleFileGPT.java generate model=model.bin "prompt=Once upon" len=300
 */
public final class SingleFileGPT {
    static final int NORM_LN = 0, NORM_RMS = 1, ACT_GELU = 0, ACT_SWIGLU = 1, POS_LEARNED = 0, POS_ROPE = 1;
    static final long MAGIC = 0x4652415947505432L; // "FRAYGPT2"
    static final long MAGIC_Q8 = 0x4652415947505138L; // "FRAYGPQ8"
    static final long MAGIC_T = 0x4652415947505433L; // "FRAYGPT3" (ternary)

    // ================================================================== PRNG
    static final class Rng {
        private long s;
        private boolean haveGauss;
        private double gauss;

        Rng(long seed) { s = seed; }

        long nextLong() {
            long z = (s += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

        int nextInt(int bound) { return (int) Long.remainderUnsigned(nextLong(), bound); }

        long nextLongBound(long bound) { return Long.remainderUnsigned(nextLong(), bound); }

        double nextGaussian() {
            if (haveGauss) { haveGauss = false; return gauss; }
            double u, v, q;
            do {
                u = 2 * nextDouble() - 1;
                v = 2 * nextDouble() - 1;
                q = u * u + v * v;
            } while (q >= 1 || q == 0);
            double f = Math.sqrt(-2 * Math.log(q) / q);
            gauss = v * f;
            haveGauss = true;
            return u * f;
        }
    }

    // ================================================================ config
    static final class Config {
        final int dim, heads, layers, ctx, vocab, norm, act, pos;

        Config(int dim, int heads, int layers, int ctx, int vocab, int norm, int act, int pos) {
            if (dim % heads != 0) throw new IllegalArgumentException("dim must be divisible by heads");
            if (pos == POS_ROPE && (dim / heads) % 2 != 0) throw new IllegalArgumentException("RoPE needs an even head size");
            if (vocab < 256) throw new IllegalArgumentException("vocab must be >= 256");
            this.dim = dim; this.heads = heads; this.layers = layers; this.ctx = ctx;
            this.vocab = vocab; this.norm = norm; this.act = act; this.pos = pos;
        }

        int hd() { return dim / heads; }

        int ff() { return act == ACT_SWIGLU ? ((int) Math.round(8.0 * dim / 3) + 7) / 8 * 8 : 4 * dim; }
    }

    // ================================================================ layout
    static final class LayerOff {
        int n1g, n1b = -1, wq, bq, wk, bk, wv, bv, wo, bo, n2g, n2b = -1, w1, b1, wu = -1, bu = -1, w2, b2;
    }

    static final class Layout {
        int p, wte, wpe = -1, nfg, nfb = -1, total;
        LayerOff[] layer;
        int[][] decay;
        private final List<int[]> dec = new ArrayList<>();
        private final boolean compact;

        private int alloc(int n, boolean decayed) { return alloc(n, decayed, 0); }

        /** cols = row width of a matrix (its output dimension); recorded so group=0 can mean one scale per row. */
        private int alloc(int n, boolean decayed, int cols) {
            int s = p;
            if (compact && decayed) return s; // matrices live in the int8/ternary store, not in the double array
            p += n;
            if (decayed) dec.add(new int[]{s, p, cols});
            return s;
        }

        Layout(Config c) { this(c, false); }

        /** compact = true: only norm gains and biases take space (the quantized inference layout). */
        Layout(Config c, boolean compact) {
            this.compact = compact;
            int D = c.dim, F = c.ff(), V = c.vocab;
            boolean ln = c.norm == NORM_LN, swi = c.act == ACT_SWIGLU;
            wte = alloc(V * D, true, D);
            if (c.pos == POS_LEARNED) wpe = alloc(c.ctx * D, true, D);
            layer = new LayerOff[c.layers];
            for (int l = 0; l < c.layers; l++) {
                LayerOff L = layer[l] = new LayerOff();
                L.n1g = alloc(D, false); if (ln) L.n1b = alloc(D, false);
                L.wq = alloc(D * D, true, D); L.bq = alloc(D, false);
                L.wk = alloc(D * D, true, D); L.bk = alloc(D, false);
                L.wv = alloc(D * D, true, D); L.bv = alloc(D, false);
                L.wo = alloc(D * D, true, D); L.bo = alloc(D, false);
                L.n2g = alloc(D, false); if (ln) L.n2b = alloc(D, false);
                L.w1 = alloc(D * F, true, F); L.b1 = alloc(F, false);
                if (swi) { L.wu = alloc(D * F, true, F); L.bu = alloc(F, false); }
                L.w2 = alloc(F * D, true, D); L.b2 = alloc(D, false);
            }
            nfg = alloc(D, false); if (ln) nfb = alloc(D, false);
            total = p;
            decay = dec.toArray(new int[0][]);
        }
    }

    // ================================================================= model
    static final class Model {
        final Config cfg;
        final Layout lay;
        final double[] w;
        QWeights qw; // non-null: matrices are read from here (lay/w then hold only norms and biases)

        Layout matLay() { return qw != null ? qw.full : lay; }
        Model(Config cfg, long seed) {
            this.cfg = cfg;
            this.lay = new Layout(cfg);
            this.w = new double[lay.total];
            init(seed);
        }

        Model(Config cfg, Layout lay, double[] w) { this.cfg = cfg; this.lay = lay; this.w = w; }

        private void init(long seed) {
            Rng r = new Rng(seed);
            int D = cfg.dim, F = cfg.ff(), V = cfg.vocab;
            double std = 0.02, resStd = std / Math.sqrt(2.0 * cfg.layers);
            fillNormal(r, lay.wte, V * D, std);
            if (lay.wpe >= 0) fillNormal(r, lay.wpe, cfg.ctx * D, std);
            for (LayerOff L : lay.layer) {
                Arrays.fill(w, L.n1g, L.n1g + D, 1.0);
                Arrays.fill(w, L.n2g, L.n2g + D, 1.0);
                fillNormal(r, L.wq, D * D, std);
                fillNormal(r, L.wk, D * D, std);
                fillNormal(r, L.wv, D * D, std);
                fillNormal(r, L.wo, D * D, resStd);
                fillNormal(r, L.w1, D * F, std);
                if (L.wu >= 0) fillNormal(r, L.wu, D * F, std);
                fillNormal(r, L.w2, F * D, resStd);
            }
            Arrays.fill(w, lay.nfg, lay.nfg + D, 1.0);
        }

        private void fillNormal(Rng r, int off, int n, double std) {
            for (int i = 0; i < n; i++) w[off + i] = r.nextGaussian() * std;
        }

        int numParams() { return lay.total; }
    }

    // ================================================================== BPE
    /** Byte-level BPE (GPT-2 style): ids 0..255 are bytes, id 256+i is merges[i]. */
    static final class Bpe {
        final int[][] merges;
        final byte[][] vocab;
        private final HashMap<Long, Integer> rank = new HashMap<>();
        private final HashMap<String, int[]> cache = new HashMap<>();

        /** Special tokens, in id order after the merges. They are matched as literal text and never split or merged. */
        static final String[] SPECIALS = {"<|endoftext|>", "<|pad|>", "<|unk|>", "<|sep|>"};
        final int nSpecial;

        Bpe(int[][] merges) { this(merges, 0); }

        Bpe(int[][] merges, int nSpecial) {
            if (nSpecial < 0 || nSpecial > SPECIALS.length) throw new IllegalArgumentException("bad special-token count " + nSpecial);
            this.merges = merges;
            this.nSpecial = nSpecial;
            vocab = new byte[256 + merges.length + nSpecial][];
            for (int k = 0; k < nSpecial; k++) vocab[256 + merges.length + k] = SPECIALS[k].getBytes(StandardCharsets.UTF_8);
            for (int i = 0; i < 256; i++) vocab[i] = new byte[]{(byte) i};
            for (int i = 0; i < merges.length; i++) {
                int a = merges[i][0], b = merges[i][1];
                if (a < 0 || b < 0 || a >= 256 + i || b >= 256 + i) throw new IllegalArgumentException("bad merge " + i);
                byte[] x = vocab[a], y = vocab[b];
                byte[] z = Arrays.copyOf(x, x.length + y.length);
                System.arraycopy(y, 0, z, x.length, y.length);
                vocab[256 + i] = z;
                rank.put(key(a, b), i);
            }
        }

        static long key(int a, int b) { return ((long) a << 32) | (b & 0xFFFFFFFFL); }

        int vocabSize() { return 256 + merges.length + nSpecial; }

        /** Id of special token k (SPECIALS order), or -1 if this tokenizer does not have it. */
        int specialId(int k) { return k < nSpecial ? 256 + merges.length + k : -1; }

        int eosId() { return specialId(0); }

        interface Seg { void text(int s, int e); void special(int id); }

        /** Walks data, reporting literal special tokens as ids and everything between them as text ranges. */
        void segments(byte[] data, int off, int end, Seg fn) {
            if (nSpecial == 0) { fn.text(off, end); return; }
            int i = off, from = off;
            while (i < end) {
                int hit = -1;
                if (data[i] == '<') {
                    for (int k = 0; k < nSpecial && hit < 0; k++) {
                        byte[] sp = vocab[256 + merges.length + k];
                        if (i + sp.length <= end && Arrays.equals(data, i, i + sp.length, sp, 0, sp.length)) hit = k;
                    }
                }
                if (hit < 0) { i++; continue; }
                if (i > from) fn.text(from, i);
                fn.special(256 + merges.length + hit);
                i += vocab[256 + merges.length + hit].length;
                from = i;
            }
            if (end > from) fn.text(from, end);
        }

        interface Piece { void accept(int s, int e); }

        static final int LETTER = 0, OTHER = 1, SPACE = 2, DIGIT = 3;

        /** Classifies the UTF-8 character at i. Returns (byteLength << 2) | class. Invalid bytes count as 1-byte letters. */
        static int charAt(byte[] b, int i, int end) {
            int c = b[i] & 0xFF;
            if (c < 0x80) {
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return (1 << 2) | LETTER;
                if (c >= '0' && c <= '9') return (1 << 2) | DIGIT;
                if (c == ' ' || (c >= 9 && c <= 13)) return (1 << 2) | SPACE;
                return (1 << 2) | OTHER;
            }
            int n = c >= 0xF0 && c < 0xF8 ? 4 : c >= 0xE0 && c < 0xF0 ? 3 : c >= 0xC2 && c < 0xE0 ? 2 : 0;
            if (n == 0 || i + n > end) return (1 << 2) | LETTER;
            int cp = c & (0xFF >> (n + 1));
            for (int k = 1; k < n; k++) {
                int cc = b[i + k] & 0xFF;
                if ((cc & 0xC0) != 0x80) return (1 << 2) | LETTER;
                cp = (cp << 6) | (cc & 0x3F);
            }
            return (n << 2) | codePointClass(cp);
        }

        static int codePointClass(int cp) {
            if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return LETTER;
            if (Character.isLetter(cp)) return LETTER;
            int t = Character.getType(cp);
            if (t == Character.DECIMAL_DIGIT_NUMBER || t == Character.LETTER_NUMBER || t == Character.OTHER_NUMBER) return DIGIT;
            if (Character.isWhitespace(cp) || Character.isSpaceChar(cp) || cp == 0x85) return SPACE;
            return OTHER;
        }

        static int contraction(byte[] b, int i, int end) {
            if (b[i] != '\'' || i + 1 >= end) return 0;
            byte c = b[i + 1];
            if (c == 's' || c == 't' || c == 'm' || c == 'd') return 2;
            if (i + 2 < end) {
                byte d = b[i + 2];
                if ((c == 'r' && d == 'e') || (c == 'v' && d == 'e') || (c == 'l' && d == 'l')) return 3;
            }
            return 0;
        }

        /**
         * GPT-2 pretokenization, equivalent to the regex
         * {@code 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+}
         * over UTF-8 text (checked against java.util.regex in selftest).
         */
        static void pieces(byte[] b, int off, int end, Piece fn) {
            int i = off;
            while (i < end) {
                int start = i;
                int k = contraction(b, i, end);
                if (k > 0) {
                    i += k;
                } else {
                    int j = b[i] == ' ' ? i + 1 : i;
                    int cj = j < end ? charAt(b, j, end) : SPACE;
                    if ((cj & 3) != SPACE) {
                        int cls = cj & 3;
                        i = j;
                        while (i < end) {
                            int ci = charAt(b, i, end);
                            if ((ci & 3) != cls) break;
                            i += ci >> 2;
                        }
                    } else {
                        int last = i, r = i;
                        while (r < end) {
                            int ci = charAt(b, r, end);
                            if ((ci & 3) != SPACE) break;
                            last = r;
                            r += ci >> 2;
                        }
                        i = (r < end && last > i) ? last : r;
                    }
                }
                fn.accept(start, i);
            }
        }

        int[] encodePiece(byte[] b, int s, int e) {
            String k = new String(b, s, e - s, StandardCharsets.ISO_8859_1);
            int[] hit = cache.get(k);
            if (hit != null) return hit;
            int[] ids = new int[e - s];
            for (int i = 0; i < ids.length; i++) ids[i] = b[s + i] & 0xFF;
            while (ids.length >= 2 && merges.length > 0) {
                int best = Integer.MAX_VALUE;
                for (int i = 0; i + 1 < ids.length; i++) {
                    Integer r = rank.get(key(ids[i], ids[i + 1]));
                    if (r != null && r < best) best = r;
                }
                if (best == Integer.MAX_VALUE) break;
                ids = mergePair(ids, merges[best][0], merges[best][1], 256 + best);
            }
            if (cache.size() < 500000) cache.put(k, ids);
            return ids;
        }

        static int[] mergePair(int[] ids, int a, int b, int nid) {
            int[] out = new int[ids.length];
            int n = 0;
            for (int i = 0; i < ids.length; ) {
                if (i + 1 < ids.length && ids[i] == a && ids[i + 1] == b) { out[n++] = nid; i += 2; }
                else out[n++] = ids[i++];
            }
            return Arrays.copyOf(out, n);
        }

        int[] encode(byte[] data) {
            ArrayList<int[]> parts = new ArrayList<>();
            int[] total = {0};
            segments(data, 0, data.length, new Seg() {
                public void text(int s0, int e0) {
                    pieces(data, s0, e0, (s, e) -> {
                        int[] p = encodePiece(data, s, e);
                        parts.add(p);
                        total[0] += p.length;
                    });
                }
                public void special(int id) { parts.add(new int[]{id}); total[0]++; }
            });
            int[] out = new int[total[0]];
            int o = 0;
            for (int[] p : parts) { System.arraycopy(p, 0, out, o, p.length); o += p.length; }
            return out;
        }

        byte[] decode(int[] ids, int n) {
            ByteArrayOutputStream o = new ByteArrayOutputStream();
            for (int i = 0; i < n; i++) o.write(vocab[ids[i]], 0, vocab[ids[i]].length);
            return o.toByteArray();
        }

        /** Reads the file in newline-aligned chunks (bounded memory). */
        static void forEachChunk(Path file, java.util.function.Consumer<byte[]> fn) throws IOException {
            try (InputStream in = new BufferedInputStream(Files.newInputStream(file), 1 << 20)) {
                byte[] carry = new byte[0], blk = new byte[1 << 22];
                int r;
                while ((r = in.read(blk)) > 0) {
                    byte[] cur = Arrays.copyOf(carry, carry.length + r);
                    System.arraycopy(blk, 0, cur, carry.length, r);
                    int cut = 0;
                    for (int i = cur.length - 1; i >= 0; i--) if (cur[i] == '\n') { cut = i + 1; break; }
                    if (cut == 0) {
                        if (cur.length > (1 << 26)) cut = cur.length; else { carry = cur; continue; }
                    }
                    fn.accept(Arrays.copyOfRange(cur, 0, cut));
                    carry = Arrays.copyOfRange(cur, cut, cur.length);
                }
                if (carry.length > 0) fn.accept(carry);
            }
        }

        /** Word-frequency table of the corpus, keyed by piece bytes (as ISO-8859-1 strings). */
        static HashMap<String, Long> wordFreq(Path file) throws IOException { return wordFreq(file, 0); }

        /** With nSpecial > 0, literal special tokens are excluded so no merges are learned across them. */
        static HashMap<String, Long> wordFreq(Path file, int nSpecial) throws IOException {
            HashMap<String, Long> freq = new HashMap<>();
            Bpe sp = new Bpe(new int[0][], nSpecial);
            forEachChunk(file, chunk -> sp.segments(chunk, 0, chunk.length, new Seg() {
                public void text(int s0, int e0) {
                    pieces(chunk, s0, e0, (s, e) -> freq.merge(new String(chunk, s, e - s, StandardCharsets.ISO_8859_1), 1L, Long::sum));
                }
                public void special(int id) { }
            }));
            return freq;
        }

        /** Trains merges until vocabSize is reached (or no pair repeats). Incremental pair counts. */
        static Bpe train(HashMap<String, Long> freq, int vocabSize) { return train(freq, vocabSize, 0); }

        /** vocabSize counts bytes + merges; the nSpecial special tokens are added on top. */
        static Bpe train(HashMap<String, Long> freq, int vocabSize, int nSpecial) {
            int nMerges = vocabSize - 256;
            List<String> keys = new ArrayList<>(freq.keySet());
            Collections.sort(keys);
            int W = keys.size();
            int[][] words = new int[W][];
            long[] cnt = new long[W];
            HashMap<Long, Long> pc = new HashMap<>();
            HashMap<Long, HashSet<Integer>> pw = new HashMap<>();
            for (int i = 0; i < W; i++) {
                String k = keys.get(i);
                int[] ids = new int[k.length()];
                for (int j = 0; j < ids.length; j++) ids[j] = k.charAt(j);
                words[i] = ids;
                cnt[i] = freq.get(k);
                for (int j = 0; j + 1 < ids.length; j++) {
                    long pk = key(ids[j], ids[j + 1]);
                    pc.merge(pk, cnt[i], Long::sum);
                    pw.computeIfAbsent(pk, z -> new HashSet<>()).add(i);
                }
            }
            List<int[]> out = new ArrayList<>();
            for (int m = 0; m < nMerges; m++) {
                long bestKey = 0, bestCount = 0;
                for (Map.Entry<Long, Long> e : pc.entrySet()) {
                    long c = e.getValue(), k = e.getKey();
                    if (c > bestCount || (c == bestCount && c > 0 && k < bestKey)) { bestCount = c; bestKey = k; }
                }
                if (bestCount <= 0) break;
                int a = (int) (bestKey >>> 32), b = (int) bestKey, nid = 256 + m;
                for (int idx : new ArrayList<>(pw.get(bestKey))) {
                    int[] old = words[idx];
                    int[] nw = mergePair(old, a, b, nid);
                    if (nw.length == old.length) continue;
                    for (int j = 0; j + 1 < old.length; j++) dec(pc, key(old[j], old[j + 1]), cnt[idx]);
                    for (int j = 0; j + 1 < nw.length; j++) {
                        long pk = key(nw[j], nw[j + 1]);
                        pc.merge(pk, cnt[idx], Long::sum);
                        pw.computeIfAbsent(pk, z -> new HashSet<>()).add(idx);
                    }
                    words[idx] = nw;
                }
                pc.remove(bestKey);
                pw.remove(bestKey);
                out.add(new int[]{a, b});
            }
            return new Bpe(out.toArray(new int[0][]), nSpecial);
        }

        private static void dec(HashMap<Long, Long> pc, long k, long by) {
            long v = pc.get(k) - by;
            if (v <= 0) pc.remove(k); else pc.put(k, v);
        }
    }

    // ============================================================= optimizer
    static final class AdamW {
        final double[] m, v;
        final boolean[] decay;
        final double b1 = 0.9, b2 = 0.95, eps = 1e-8, wd;
        int step = 0;

        AdamW(Layout lay, double wd) {
            m = new double[lay.total]; v = new double[lay.total];
            decay = new boolean[lay.total];
            for (int[] r : lay.decay) Arrays.fill(decay, r[0], r[1], true);
            this.wd = wd;
        }

        void update(double[] w, double[] g, double lr) {
            step++;
            double c1 = 1 - Math.pow(b1, step), c2 = 1 - Math.pow(b2, step);
            for (int i = 0; i < w.length; i++) {
                m[i] = b1 * m[i] + (1 - b1) * g[i];
                v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
                double upd = (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
                if (decay[i]) upd += wd * w[i];
                w[i] -= lr * upd;
            }
        }
    }

    static double lrAt(int step, int total, double base) {
        int warm = Math.max(1, Math.min(100, total / 10));
        if (step < warm) return base * (step + 1) / warm;
        double prog = (double) (step - warm) / Math.max(1, total - warm);
        double min = base * 0.1;
        return min + 0.5 * (base - min) * (1 + Math.cos(Math.PI * prog));
    }

    // ============================================================ checkpoints
    static final class Loaded {
        Model model;
        Bpe bpe;
        AdamW opt; // null when the checkpoint carries no optimizer state
    }

    static void saveCkpt(String path, Model m, Bpe bpe, AdamW opt) throws IOException {
        Path tmp = Paths.get(path + ".tmp");
        try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tmp), 1 << 20))) {
            Config c = m.cfg;
            o.writeLong(MAGIC);
            o.writeInt(c.dim); o.writeInt(c.heads); o.writeInt(c.layers); o.writeInt(c.ctx);
            o.writeInt(c.vocab); o.writeInt(c.norm); o.writeInt(c.act); o.writeInt(c.pos);
            o.writeInt(bpe.merges.length);
            for (int[] p : bpe.merges) { o.writeInt(p[0]); o.writeInt(p[1]); }
            o.writeInt(m.w.length);
            for (double x : m.w) o.writeDouble(x);
            o.writeInt(opt != null ? 1 : 0);
            if (opt != null) {
                o.writeInt(opt.step);
                o.writeDouble(opt.wd);
                for (double x : opt.m) o.writeDouble(x);
                for (double x : opt.v) o.writeDouble(x);
            }
        }
        Files.move(tmp, Paths.get(path), StandardCopyOption.REPLACE_EXISTING);
    }

    static Loaded loadCkpt(String path) throws IOException { return loadCkpt(path, false); }

    /** resident = true keeps int8/ternary matrices as bytes (about 8x less memory than doubles). */
    static Loaded loadCkpt(String path, boolean resident) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(Paths.get(path)), 1 << 20))) {
            long magic = in.readLong();
            if (magic != MAGIC && magic != MAGIC_Q8 && magic != MAGIC_T) throw new IOException("Not a FraynixGPT v2 checkpoint");
            Config c = new Config(in.readInt(), in.readInt(), in.readInt(), in.readInt(),
                    in.readInt(), in.readInt(), in.readInt(), in.readInt());
            int nm = in.readInt();
            int[][] merges = new int[nm][2];
            for (int i = 0; i < nm; i++) { merges[i][0] = in.readInt(); merges[i][1] = in.readInt(); }
            Layout lay = new Layout(c);
            if (magic != MAGIC) {
                Loaded L = new Loaded();
                L.model = resident ? readResident(in, c, lay, magic == MAGIC_T)
                        : new Model(c, lay, magic == MAGIC_T ? readTernary(in, lay) : readQuant(in, lay));
                L.bpe = new Bpe(merges, c.vocab - 256 - nm);
                if (L.bpe.vocabSize() != c.vocab) throw new IOException("Tokenizer/vocab mismatch");
                return L;
            }
            int n = in.readInt();
            if (n != lay.total) throw new IOException("Checkpoint size mismatch");
            double[] w = new double[n];
            for (int i = 0; i < n; i++) w[i] = in.readDouble();
            Loaded L = new Loaded();
            L.model = new Model(c, lay, w);
            L.bpe = new Bpe(merges, c.vocab - 256 - nm);
            if (L.bpe.vocabSize() != c.vocab) throw new IOException("Tokenizer/vocab mismatch");
            if (in.readInt() == 1) {
                int step = in.readInt();
                AdamW a = new AdamW(lay, in.readDouble());
                a.step = step;
                for (int i = 0; i < n; i++) a.m[i] = in.readDouble();
                for (int i = 0; i < n; i++) a.v[i] = in.readDouble();
                L.opt = a;
            }
            return L;
        }
    }

    // ======================================================== int8 quantization
    /**
     * Weight-only int8, symmetric, one float scale per `group` consecutive weights. Only the matrices (the
     * weight-decayed ranges: embeddings and linear weights) are quantized; norm gains and biases stay exact doubles.
     * No Adam state is stored: this is an inference format. Loading dequantizes to doubles.
     */
    /** Scale-group size for matrix range r: `group` consecutive weights, or (group = 0) one scale per matrix row. */
    static int gs(int[] r, int group) { return group > 0 ? group : r[2]; }

    static boolean[] matrixMask(Layout lay) {
        boolean[] mat = new boolean[lay.total];
        for (int[] r : lay.decay) for (int i = r[0]; i < r[1]; i++) mat[i] = true;
        return mat;
    }

    static void saveQuant(String path, Model m, Bpe bpe, int group) throws IOException {
        Path tmp = Paths.get(path + ".tmp");
        try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tmp), 1 << 20))) {
            Config c = m.cfg;
            o.writeLong(MAGIC_Q8);
            o.writeInt(c.dim); o.writeInt(c.heads); o.writeInt(c.layers); o.writeInt(c.ctx);
            o.writeInt(c.vocab); o.writeInt(c.norm); o.writeInt(c.act); o.writeInt(c.pos);
            o.writeInt(bpe.merges.length);
            for (int[] p : bpe.merges) { o.writeInt(p[0]); o.writeInt(p[1]); }
            o.writeInt(m.w.length);
            o.writeInt(group);
            for (int[] r : m.lay.decay)
                for (int g = r[0]; g < r[1]; g += gs(r, group)) {
                    int end = Math.min(r[1], g + gs(r, group));
                    double max = 0;
                    for (int i = g; i < end; i++) max = Math.max(max, Math.abs(m.w[i]));
                    float scale = (float) (max / 127.0);
                    o.writeFloat(scale);
                    for (int i = g; i < end; i++) {
                        int q = scale == 0 ? 0 : (int) Math.round(m.w[i] / (double) scale);
                        o.writeByte(Math.max(-127, Math.min(127, q)));
                    }
                }
            boolean[] mat = matrixMask(m.lay);
            for (int i = 0; i < m.w.length; i++) if (!mat[i]) o.writeDouble(m.w[i]);
        }
        Files.move(tmp, Paths.get(path), StandardCopyOption.REPLACE_EXISTING);
    }

    static double[] readQuant(DataInputStream in, Layout lay) throws IOException {
        int n = in.readInt();
        if (n != lay.total) throw new IOException("Checkpoint size mismatch");
        int group = in.readInt();
        if (group < 0) throw new IOException("Bad quantization group");
        double[] w = new double[n];
        for (int[] r : lay.decay)
            for (int g = r[0]; g < r[1]; g += gs(r, group)) {
                int end = Math.min(r[1], g + gs(r, group));
                double scale = in.readFloat();
                for (int i = g; i < end; i++) w[i] = in.readByte() * scale;
            }
        boolean[] mat = matrixMask(lay);
        for (int i = 0; i < n; i++) if (!mat[i]) w[i] = in.readDouble();
        return w;
    }

    // ================================================== ternary weights {-1, 0, +1}
    /** BitNet b1.58 absmean rule: scale = mean|w| over the group, q = clamp(round(w / scale), -1, 1). */
    static float tScale(double[] w, int g, int end) {
        double s = 0;
        for (int i = g; i < end; i++) s += Math.abs(w[i]);
        return (float) (s / (end - g));
    }

    static int tq(double x, float s) {
        if (s == 0) return 0;
        return (int) Math.max(-1, Math.min(1, Math.round(x / s)));
    }

    /** dst = src with every matrix weight replaced by (ternary value) x (group scale); other params copied. */
    static void ternarize(double[] src, double[] dst, Layout lay, int group) {
        System.arraycopy(src, 0, dst, 0, src.length);
        for (int[] r : lay.decay)
            for (int g = r[0]; g < r[1]; g += gs(r, group)) {
                int end = Math.min(r[1], g + gs(r, group));
                float s = tScale(src, g, end);
                for (int i = g; i < end; i++) dst[i] = tq(src[i], s) * (double) s;
            }
    }

    /** 5 trits per byte (3^5 = 243 <= 256), i.e. 1.6 bits per weight, plus one float scale per group. */
    static void saveTernary(String path, Model m, Bpe bpe, int group) throws IOException {
        Path tmp = Paths.get(path + ".tmp");
        try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tmp), 1 << 20))) {
            Config c = m.cfg;
            o.writeLong(MAGIC_T);
            o.writeInt(c.dim); o.writeInt(c.heads); o.writeInt(c.layers); o.writeInt(c.ctx);
            o.writeInt(c.vocab); o.writeInt(c.norm); o.writeInt(c.act); o.writeInt(c.pos);
            o.writeInt(bpe.merges.length);
            for (int[] p : bpe.merges) { o.writeInt(p[0]); o.writeInt(p[1]); }
            o.writeInt(m.w.length);
            o.writeInt(group);
            for (int[] r : m.lay.decay)
                for (int g = r[0]; g < r[1]; g += gs(r, group)) {
                    int end = Math.min(r[1], g + gs(r, group));
                    float s = tScale(m.w, g, end);
                    o.writeFloat(s);
                    for (int i = g; i < end; i += 5) {
                        int v = 0;
                        for (int j = 0; j < 5; j++) v = v * 3 + (i + j < end ? tq(m.w[i + j], s) + 1 : 1);
                        o.writeByte(v);
                    }
                }
            boolean[] mat = matrixMask(m.lay);
            for (int i = 0; i < m.w.length; i++) if (!mat[i]) o.writeDouble(m.w[i]);
        }
        Files.move(tmp, Paths.get(path), StandardCopyOption.REPLACE_EXISTING);
    }

    static double[] readTernary(DataInputStream in, Layout lay) throws IOException {
        int n = in.readInt();
        if (n != lay.total) throw new IOException("Checkpoint size mismatch");
        int group = in.readInt();
        if (group < 0) throw new IOException("Bad quantization group");
        double[] w = new double[n];
        for (int[] r : lay.decay)
            for (int g = r[0]; g < r[1]; g += gs(r, group)) {
                int end = Math.min(r[1], g + gs(r, group));
                double s = in.readFloat();
                for (int i = g; i < end; i += 5) {
                    int v = in.readUnsignedByte();
                    if (v >= 243) throw new IOException("Corrupt ternary block");
                    int[] d = new int[5];
                    for (int j = 4; j >= 0; j--) { d[j] = v % 3; v /= 3; }
                    for (int j = 0; j < 5 && i + j < end; j++) w[i + j] = (d[j] - 1) * s;
                }
            }
        boolean[] mat = matrixMask(lay);
        for (int i = 0; i < n; i++) if (!mat[i]) w[i] = in.readDouble();
        return w;
    }

    /** Matrices stored as one byte per weight (int8, or -1/0/+1) plus a float scale per group; used for inference. */
    static final class QWeights {
        final Layout full;
        final int group;
        final byte[] q;
        final float[] scale;
        private final HashMap<Integer, Integer> base = new HashMap<>();
        private final HashMap<Integer, Integer> gsz = new HashMap<>();

        QWeights(Layout full, int group) {
            this.full = full; this.group = group;
            q = new byte[full.total];
            int cnt = 0;
            for (int[] r : full.decay) { int s = gs(r, group); base.put(r[0], cnt); gsz.put(r[0], s); cnt += (r[1] - r[0] + s - 1) / s; }
            scale = new float[cnt];
        }

        int sb(int matrixStart) { return base.get(matrixStart); }

        /** Scale-group size of the matrix starting at matrixStart. */
        int grp(int matrixStart) { return gsz.get(matrixStart); }
    }

    static void lin(Model m, double[] x, int mo, int bo, double[] y, int from, int n, int in, int out) {
        int work = (n - from) * in * out;
        if (m.qw == null) parCols(out, work, (o0, o1) -> { linearR(x, m.w, mo, bo, y, from, n, in, out, o0, o1); return 0; });
        else parCols(out, work, (o0, o1) -> { linearQR(x, m.qw, mo, m.w, bo, y, from, n, in, out, o0, o1); return 0; });
    }

    /** y[from..n) = x W + b with W dequantized on the fly (scale applied once per group segment). */
    static void linearQ(double[] x, QWeights qw, int ms, double[] w, int bo, double[] y, int from, int n, int in, int out) {
        linearQR(x, qw, ms, w, bo, y, from, n, in, out, 0, out);
    }

    static void linearQR(double[] x, QWeights qw, int ms, double[] w, int bo, double[] y, int from, int n, int in, int out, int o0, int o1) {
        byte[] q = qw.q;
        float[] sc = qw.scale;
        int grp = qw.grp(ms), sb = qw.sb(ms);
        for (int t = from; t < n; t++) {
            int yb = t * out, xb = t * in;
            for (int o = o0; o < o1; o++) y[yb + o] = w[bo + o];
            if (grp % out == 0) {
                // The scale is constant along a row, so it folds into the activation and the inner loop is a plain
                // float axpy over bytes (auto-vectorized). Partial sums are float over blocks of 64 rows, then
                // flushed into the double output, so the result is within ~1e-6 relative of the exact path.
                float[] acc = new float[out];
                for (int i0 = 0; i0 < in; i0 += 64) {
                    Arrays.fill(acc, o0, o1, 0f);
                    int i1 = Math.min(in, i0 + 64);
                    for (int i = i0; i < i1; i++) {
                        float xs = (float) (x[xb + i] * sc[sb + (i * out) / grp]);
                        int qb = ms + i * out;
                        for (int o = o0; o < o1; o++) acc[o] += xs * q[qb + o];
                    }
                    for (int o = o0; o < o1; o++) y[yb + o] += acc[o];
                }
                continue;
            }
            for (int i = 0; i < in; i++) {
                double xi = x[xb + i];
                int eo = i * out, o = o0;
                while (o < o1) {
                    int g = (eo + o) / grp;
                    int segEnd = Math.min(o1, (g + 1) * grp - eo);
                    double xs = xi * sc[sb + g];
                    int qb = ms + eo;
                    for (; o < segEnd; o++) y[yb + o] += xs * q[qb + o];
                }
            }
        }
    }

    /** dst[do..do+len) (+)= dequantized matrix elements [eo, eo+len). */
    static void qRow(QWeights qw, int ms, int eo, int len, double[] dst, int dOff, boolean add) {
        int sb = qw.sb(ms), grp = qw.grp(ms);
        for (int j = 0; j < len; j++) {
            double v = qw.q[ms + eo + j] * (double) qw.scale[sb + (eo + j) / grp];
            dst[dOff + j] = add ? dst[dOff + j] + v : v;
        }
    }

    /** sum_j x[xo+j] * dequantized matrix element [eo+j]. */
    static double qDot(QWeights qw, int ms, int sb, int eo, int len, double[] x, int xo) {
        byte[] q = qw.q;
        int grp = qw.grp(ms);
        double acc = 0;
        int j = 0;
        while (j < len) {
            int g = (eo + j) / grp;
            int segEnd = Math.min(len, (g + 1) * grp - eo);
            double s = 0;
            int qb = ms + eo;
            for (; j < segEnd; j++) s += x[xo + j] * q[qb + j];
            acc += s * qw.scale[sb + g];
        }
        return acc;
    }

    /** Builds a byte-resident inference model from a double model (group scales, int8 or ternary). */
    static Model residentFrom(Model m, int group, boolean tern) {
        Layout full = m.lay;
        QWeights qw = new QWeights(full, group);
        for (int[] r : full.decay) {
            int b = qw.sb(r[0]), gi = 0;
            for (int g = r[0]; g < r[1]; g += gs(r, group), gi++) {
                int end = Math.min(r[1], g + gs(r, group));
                float s;
                if (tern) {
                    s = tScale(m.w, g, end);
                    for (int i = g; i < end; i++) qw.q[i] = (byte) tq(m.w[i], s);
                } else {
                    double mx = 0;
                    for (int i = g; i < end; i++) mx = Math.max(mx, Math.abs(m.w[i]));
                    s = (float) (mx / 127.0);
                    for (int i = g; i < end; i++) {
                        int v = s == 0 ? 0 : (int) Math.round(m.w[i] / (double) s);
                        qw.q[i] = (byte) Math.max(-127, Math.min(127, v));
                    }
                }
                qw.scale[b + gi] = s;
            }
        }
        Layout cl = new Layout(m.cfg, true);
        double[] aux = new double[cl.total];
        boolean[] mat = matrixMask(full);
        int k = 0;
        for (int i = 0; i < mat.length; i++) if (!mat[i]) aux[k++] = m.w[i];
        Model r = new Model(m.cfg, cl, aux);
        r.qw = qw;
        return r;
    }

    static Model readResident(DataInputStream in, Config c, Layout full, boolean tern) throws IOException {
        int n = in.readInt();
        if (n != full.total) throw new IOException("Checkpoint size mismatch");
        int group = in.readInt();
        if (group < 0) throw new IOException("Bad quantization group");
        QWeights qw = new QWeights(full, group);
        for (int[] r : full.decay) {
            int b = qw.sb(r[0]), gi = 0;
            for (int g = r[0]; g < r[1]; g += gs(r, group), gi++) {
                int end = Math.min(r[1], g + gs(r, group));
                qw.scale[b + gi] = in.readFloat();
                if (!tern) {
                    in.readFully(qw.q, g, end - g);
                } else {
                    for (int i = g; i < end; i += 5) {
                        int v = in.readUnsignedByte();
                        if (v >= 243) throw new IOException("Corrupt ternary block");
                        int[] d = new int[5];
                        for (int j = 4; j >= 0; j--) { d[j] = v % 3; v /= 3; }
                        for (int j = 0; j < 5 && i + j < end; j++) qw.q[i + j] = (byte) (d[j] - 1);
                    }
                }
            }
        }
        Layout cl = new Layout(c, true);
        double[] aux = new double[cl.total];
        boolean[] mat = matrixMask(full);
        int k = 0;
        for (int i = 0; i < mat.length; i++) if (!mat[i]) aux[k++] = in.readDouble();
        Model r = new Model(c, cl, aux);
        r.qw = qw;
        return r;
    }

    static void quantizeCmd(Map<String, String> a) throws Exception {
        String in = need(a, "model"), out = need(a, "out");
        boolean tern = "ternary".equals(a.getOrDefault("bits", "8"));
        int group = geti(a, "group", 64);
        Loaded L = loadCkpt(in);
        if (tern) saveTernary(out, L.model, L.bpe, group); else saveQuant(out, L.model, L.bpe, group);
        Loaded Q = loadCkpt(out);
        double maxErr = 0, sq = 0, sw = 0;
        boolean[] mat = matrixMask(L.model.lay);
        long cnt = 0;
        for (int i = 0; i < mat.length; i++) if (mat[i]) {
            double d = L.model.w[i] - Q.model.w[i];
            maxErr = Math.max(maxErr, Math.abs(d)); sq += d * d; sw += L.model.w[i] * L.model.w[i]; cnt++;
        }
        System.out.printf("size: %,d -> %,d bytes (%.2fx smaller)%n", Files.size(Paths.get(in)), Files.size(Paths.get(out)),
                (double) Files.size(Paths.get(in)) / Files.size(Paths.get(out)));
        System.out.printf("weight error: max %.3e, relative RMS %.3e over %,d matrix weights%n", maxErr, Math.sqrt(sq / sw), cnt);
        if (a.containsKey("data")) {
            byte[] raw = Files.readAllBytes(Paths.get(a.get("data")));
            Corpus cp = new MemTokens(L.bpe.encode(raw));
            int windows = geti(a, "windows", 20);
            double l0 = evalLoss(L.model, new Cache(L.model.cfg), cp, 0, cp.length(), 7, windows);
            double l1 = evalLoss(Q.model, new Cache(Q.model.cfg), cp, 0, cp.length(), 7, windows);
            System.out.printf("loss on %d windows: original %.4f, quantized %.4f (perplexity %.2f -> %.2f)%n", windows, l0, l1, Math.exp(l0), Math.exp(l1));
        }
    }

    // ================================================================ caches
    static final class LayerCache {
        double[] xin, xhat1, rstd1, ln1o, q, k, v, p, att, xmid, xhat2, rstd2, ln2o, hpre, upre, hact, xout, m1, m2;

        LayerCache(int T, int D, int F, int H) {
            xhat1 = new double[T * D]; rstd1 = new double[T]; ln1o = new double[T * D];
            q = new double[T * D]; k = new double[T * D]; v = new double[T * D];
            p = new double[H * (T * (T + 1) / 2)]; att = new double[T * D];
            xmid = new double[T * D];
            xhat2 = new double[T * D]; rstd2 = new double[T]; ln2o = new double[T * D];
            hpre = new double[T * F]; upre = new double[T * F]; hact = new double[T * F];
            xout = new double[T * D];
            m1 = new double[T * D]; m2 = new double[T * D];
        }
    }

    static final class Cache {
        final int T, D, F, H, hd, V;
        final double[] x0, xhatF, rstdF, lnfo, probs, ropeCos, ropeSin;
        final LayerCache[] lc;
        final double[] dx, dtmp, dhact, dhpre, dhup, dln2, datt, dq, dk, dv, dln1, dlnf;

        Cache(Config c) {
            T = c.ctx; D = c.dim; F = c.ff(); H = c.heads; hd = c.hd(); V = c.vocab;
            x0 = new double[T * D];
            xhatF = new double[T * D]; rstdF = new double[T]; lnfo = new double[T * D];
            probs = new double[T * V];
            lc = new LayerCache[c.layers];
            for (int l = 0; l < c.layers; l++) lc[l] = new LayerCache(T, D, F, H);
            dx = new double[T * D]; dtmp = new double[T * D];
            dhact = new double[T * F]; dhpre = new double[T * F]; dhup = new double[T * F];
            dln2 = new double[T * D]; datt = new double[T * D];
            dq = new double[T * D]; dk = new double[T * D]; dv = new double[T * D];
            dln1 = new double[T * D]; dlnf = new double[T * D];
            int half = hd / 2;
            ropeCos = new double[T * half]; ropeSin = new double[T * half];
            for (int t = 0; t < T; t++)
                for (int i = 0; i < half; i++) {
                    double ang = t * Math.pow(10000.0, -2.0 * i / hd);
                    ropeCos[t * half + i] = Math.cos(ang);
                    ropeSin[t * half + i] = Math.sin(ang);
                }
        }
    }

    // ============================================================= primitives
    /** y[from..n) = x W + b, with W stored [in][out]. */
    static void linear(double[] x, double[] w, int wo, int bo, double[] y, int from, int n, int in, int out) {
        linearR(x, w, wo, bo, y, from, n, in, out, 0, out);
    }

    /** Output columns [o0,o1) only; each element's summation order does not depend on the range. */
    static void linearR(double[] x, double[] w, int wo, int bo, double[] y, int from, int n, int in, int out, int o0, int o1) {
        for (int t = from; t < n; t++) {
            int yb = t * out, xb = t * in;
            for (int o = o0; o < o1; o++) y[yb + o] = w[bo + o];
            int i = 0;
            for (; i + 3 < in; i += 4) {
                double x0 = x[xb + i], x1 = x[xb + i + 1], x2 = x[xb + i + 2], x3 = x[xb + i + 3];
                int w0 = wo + i * out, w1 = w0 + out, w2 = w1 + out, w3 = w2 + out;
                for (int o = o0; o < o1; o++)
                    y[yb + o] += x0 * w[w0 + o] + x1 * w[w1 + o] + x2 * w[w2 + o] + x3 * w[w3 + o];
            }
            for (; i < in; i++) {
                double xi = x[xb + i];
                int wb = wo + i * out;
                for (int o = o0; o < o1; o++) y[yb + o] += xi * w[wb + o];
            }
        }
    }

    /** Inference-only parallelism (generate/eval/bench set it); training and selftest keep 1 to avoid nesting pools. */
    static int inferThreads = 1;
    static final java.util.concurrent.atomic.AtomicLong parCalls = new java.util.concurrent.atomic.AtomicLong();

    /**
     * Persistent worker pool for decoding. Workers spin on a generation counter (microsecond dispatch, unlike
     * fork-join wake-ups) and park after a short idle spin so an idle process does not burn CPU.
     */
    static final class SpinPool {
        final int T;
        final Thread[] th;
        volatile int gen;
        volatile boolean stop;
        volatile java.util.function.IntConsumer task;
        volatile Throwable error;
        final java.util.concurrent.atomic.AtomicInteger remaining = new java.util.concurrent.atomic.AtomicInteger();

        SpinPool(int T) {
            this.T = T;
            th = new Thread[T - 1];
            for (int k = 1; k < T; k++) {
                final int id = k;
                Thread t = new Thread(() -> work(id), "fraynix-spin-" + k);
                t.setDaemon(true);
                th[k - 1] = t;
                t.start();
            }
        }

        private void work(int id) {
            int seen = 0, idle = 0;
            while (!stop) {
                int g = gen;
                if (g == seen) {
                    if (++idle < 50_000) Thread.onSpinWait();
                    else java.util.concurrent.locks.LockSupport.parkNanos(100_000);
                    continue;
                }
                idle = 0;
                seen = g;
                try { task.accept(id); } catch (Throwable e) { error = e; }
                remaining.decrementAndGet();
            }
        }

        synchronized void run(java.util.function.IntConsumer f) {
            error = null;
            task = f;
            remaining.set(T - 1);
            gen++;
            for (Thread t : th) java.util.concurrent.locks.LockSupport.unpark(t);
            try { f.accept(0); } catch (Throwable e) { error = e; }
            while (remaining.get() > 0) Thread.onSpinWait();
            if (error != null) throw new RuntimeException(error);
        }

        void close() { stop = true; for (Thread t : th) java.util.concurrent.locks.LockSupport.unpark(t); }
    }

    static SpinPool pool;

    static synchronized SpinPool poolFor(int T) {
        if (pool == null || pool.T != T) { if (pool != null) pool.close(); pool = new SpinPool(T); }
        return pool;
    }

    static void parCols(int out, int work, java.util.function.IntBinaryOperator body) {
        int T = inferThreads;
        if (T <= 1 || work < (1 << 15) || out < 2 * T) { body.applyAsInt(0, out); return; }
        parCalls.incrementAndGet();
        // contiguous slices on 64-column boundaries: no cache line is shared between threads
        poolFor(T).run(k -> body.applyAsInt(bound(out, k, T), bound(out, k + 1, T)));
    }

    /** Column-slice alignment for parallel matmuls (doubles per cache line = 8; bytes per line = 64). Tunable: -Dfray.align=N (power of 2). */
    static final int ALIGN = Integer.getInteger("fray.align", 64);

    static int bound(int out, int k, int T) {
        if (k >= T) return out;
        long b = (long) out * k / T;
        return out >= (long) ALIGN * T ? (int) (b & ~(long) (ALIGN - 1)) : (int) b;
    }

    /** Accumulates into g and (if non-null) dx. */
    static void linearBack(double[] x, double[] dy, double[] dx, double[] w, double[] g,
                           int wo, int bo, int n, int in, int out) {
        for (int t = 0; t < n; t++) {
            int yb = t * out;
            for (int i = 0; i < in; i++) {
                double xi = x[t * in + i], s = 0;
                int wb = wo + i * out;
                for (int o = 0; o < out; o++) {
                    double d = dy[yb + o];
                    g[wb + o] += xi * d;
                    s += w[wb + o] * d;
                }
                if (dx != null) dx[t * in + i] += s;
            }
            for (int o = 0; o < out; o++) g[bo + o] += dy[yb + o];
        }
    }

    static void norm(int kind, double[] x, double[] w, int go, int bo, double[] y, double[] xhat, double[] rstd,
                     int from, int n, int D) {
        for (int t = from; t < n; t++) {
            int b = t * D;
            if (kind == NORM_LN) {
                double mean = 0;
                for (int i = 0; i < D; i++) mean += x[b + i];
                mean /= D;
                double var = 0;
                for (int i = 0; i < D; i++) { double d = x[b + i] - mean; var += d * d; }
                double rs = 1.0 / Math.sqrt(var / D + 1e-5);
                rstd[t] = rs;
                for (int i = 0; i < D; i++) {
                    double h = (x[b + i] - mean) * rs;
                    xhat[b + i] = h;
                    y[b + i] = h * w[go + i] + w[bo + i];
                }
            } else {
                double ms = 0;
                for (int i = 0; i < D; i++) ms += x[b + i] * x[b + i];
                double rs = 1.0 / Math.sqrt(ms / D + 1e-5);
                rstd[t] = rs;
                for (int i = 0; i < D; i++) {
                    double h = x[b + i] * rs;
                    xhat[b + i] = h;
                    y[b + i] = h * w[go + i];
                }
            }
        }
    }

    /** Accumulates into dx and g. */
    static void normBack(int kind, double[] xhat, double[] rstd, double[] w, double[] g, int go, int bo,
                         double[] dy, double[] dx, int n, int D) {
        for (int t = 0; t < n; t++) {
            int b = t * D;
            double m1 = 0, m2 = 0;
            for (int i = 0; i < D; i++) {
                double dh = dy[b + i] * w[go + i];
                m1 += dh;
                m2 += dh * xhat[b + i];
                g[go + i] += dy[b + i] * xhat[b + i];
                if (kind == NORM_LN) g[bo + i] += dy[b + i];
            }
            m1 /= D; m2 /= D;
            if (kind == NORM_RMS) m1 = 0;
            for (int i = 0; i < D; i++) {
                double dh = dy[b + i] * w[go + i];
                dx[b + i] += rstd[t] * (dh - m1 - xhat[b + i] * m2);
            }
        }
    }

    static final double GC = Math.sqrt(2.0 / Math.PI);

    static double gelu(double x) {
        double u = GC * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1 + Math.tanh(u));
    }

    static double geluGrad(double x) {
        double u = GC * (x + 0.044715 * x * x * x);
        double th = Math.tanh(u);
        return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * GC * (1 + 3 * 0.044715 * x * x);
    }

    static double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }

    static double silu(double x) { return x * sigmoid(x); }

    static double siluGrad(double x) {
        double s = sigmoid(x);
        return s * (1 + x * (1 - s));
    }

    /** Rotary embedding (rotate-half convention) applied in place to rows [from,n). */
    static void ropeApply(double[] x, double[] cos, double[] sin, int from, int n, int D, int H) {
        int hd = D / H, half = hd / 2;
        for (int t = from; t < n; t++)
            for (int h = 0; h < H; h++)
                for (int i = 0; i < half; i++) {
                    int i1 = t * D + h * hd + i, i2 = i1 + half;
                    double c = cos[t * half + i], s = sin[t * half + i], a = x[i1], b = x[i2];
                    x[i1] = a * c - b * s;
                    x[i2] = a * s + b * c;
                }
    }

    /** Inverse rotation: maps gradients w.r.t. rotated vectors back to the unrotated ones. */
    static void ropeBack(double[] dx, double[] cos, double[] sin, int n, int D, int H) {
        int hd = D / H, half = hd / 2;
        for (int t = 0; t < n; t++)
            for (int h = 0; h < H; h++)
                for (int i = 0; i < half; i++) {
                    int i1 = t * D + h * hd + i, i2 = i1 + half;
                    double c = cos[t * half + i], s = sin[t * half + i], a = dx[i1], b = dx[i2];
                    dx[i1] = a * c + b * s;
                    dx[i2] = -a * s + b * c;
                }
    }

    static int pbase(int h, int t, int T) { return h * (T * (T + 1) / 2) + t * (t + 1) / 2; }

    /** Causal attention for rows [from,n); earlier rows' k/v are read from the cache arrays. */
    static void attnForward(LayerCache a, int from, int n, int T, int D, int H) {
        int hd = D / H;
        double scale = 1.0 / Math.sqrt(hd);
        for (int h = 0; h < H; h++) {
            int ho = h * hd;
            for (int t = from; t < n; t++) {
                int pb = pbase(h, t, T);
                double max = Double.NEGATIVE_INFINITY;
                for (int s = 0; s <= t; s++) {
                    double dot = 0;
                    for (int j = 0; j < hd; j++) dot += a.q[t * D + ho + j] * a.k[s * D + ho + j];
                    dot *= scale;
                    a.p[pb + s] = dot;
                    if (dot > max) max = dot;
                }
                double sum = 0;
                for (int s = 0; s <= t; s++) { double e = Math.exp(a.p[pb + s] - max); a.p[pb + s] = e; sum += e; }
                for (int s = 0; s <= t; s++) a.p[pb + s] /= sum;
                for (int j = 0; j < hd; j++) {
                    double acc = 0;
                    for (int s = 0; s <= t; s++) acc += a.p[pb + s] * a.v[s * D + ho + j];
                    a.att[t * D + ho + j] = acc;
                }
            }
        }
    }

    /** Accumulates into dq, dk, dv (which the caller zeroes). */
    static void attnBack(LayerCache a, double[] datt, double[] dq, double[] dk, double[] dv, int n, int T, int D, int H) {
        int hd = D / H;
        double scale = 1.0 / Math.sqrt(hd);
        double[] dp = new double[T];
        for (int h = 0; h < H; h++) {
            int ho = h * hd;
            for (int t = 0; t < n; t++) {
                int pb = pbase(h, t, T);
                double dot = 0;
                for (int s = 0; s <= t; s++) {
                    double d = 0;
                    for (int j = 0; j < hd; j++) d += datt[t * D + ho + j] * a.v[s * D + ho + j];
                    dp[s] = d;
                    dot += a.p[pb + s] * d;
                    for (int j = 0; j < hd; j++) dv[s * D + ho + j] += a.p[pb + s] * datt[t * D + ho + j];
                }
                for (int s = 0; s <= t; s++) {
                    double ds = a.p[pb + s] * (dp[s] - dot) * scale;
                    for (int j = 0; j < hd; j++) {
                        dq[t * D + ho + j] += ds * a.k[s * D + ho + j];
                        dk[s * D + ho + j] += ds * a.q[t * D + ho + j];
                    }
                }
            }
        }
    }

    static void genMask(double[] m, Rng r, double p, int from, int n, int D) {
        double keep = 1 - p, scale = 1 / keep;
        for (int i = from * D; i < n * D; i++) m[i] = r.nextDouble() < keep ? scale : 0;
    }

    // ================================================================ forward
    /**
     * Runs the transformer on rows [from,n) of tok; rows below `from` must already be in the cache
     * (this is the KV cache: generation calls forward with from = number of tokens already processed).
     * With targets (from must be 0) returns the mean cross-entropy and leaves probabilities for every row
     * in c.probs; otherwise only the last row's probabilities are computed.
     * Dropout (probability dp, masks drawn from `drop`) is applied to both residual branches.
     */
    static double forward(Model m, Cache c, int[] tok, int[] tgt, int from, int n, Rng drop, double dp) {
        Config cfg = m.cfg; Layout lay = m.lay; double[] w = m.w; Layout ml = m.matLay(); QWeights qw = m.qw;
        int D = cfg.dim, F = cfg.ff(), H = cfg.heads, T = cfg.ctx, V = cfg.vocab;
        boolean rope = cfg.pos == POS_ROPE, swi = cfg.act == ACT_SWIGLU, useDrop = dp > 0 && drop != null;

        for (int t = from; t < n; t++) {
            if (qw != null) {
                qRow(qw, ml.wte, tok[t] * D, D, c.x0, t * D, false);
                if (ml.wpe >= 0) qRow(qw, ml.wpe, t * D, D, c.x0, t * D, true);
                continue;
            }
            int e = lay.wte + tok[t] * D;
            for (int d = 0; d < D; d++) c.x0[t * D + d] = w[e + d];
            if (lay.wpe >= 0) for (int d = 0; d < D; d++) c.x0[t * D + d] += w[lay.wpe + t * D + d];
        }
        double[] cur = c.x0;
        for (int l = 0; l < cfg.layers; l++) {
            LayerOff L = lay.layer[l], LM = ml.layer[l];
            LayerCache a = c.lc[l];
            a.xin = cur;
            norm(cfg.norm, a.xin, w, L.n1g, L.n1b, a.ln1o, a.xhat1, a.rstd1, from, n, D);
            lin(m, a.ln1o, LM.wq, L.bq, a.q, from, n, D, D);
            lin(m, a.ln1o, LM.wk, L.bk, a.k, from, n, D, D);
            lin(m, a.ln1o, LM.wv, L.bv, a.v, from, n, D, D);
            if (rope) {
                ropeApply(a.q, c.ropeCos, c.ropeSin, from, n, D, H);
                ropeApply(a.k, c.ropeCos, c.ropeSin, from, n, D, H);
            }
            attnForward(a, from, n, T, D, H);
            lin(m, a.att, LM.wo, L.bo, a.xmid, from, n, D, D);
            if (useDrop) genMask(a.m1, drop, dp, from, n, D);
            for (int i = from * D; i < n * D; i++) a.xmid[i] = (useDrop ? a.xmid[i] * a.m1[i] : a.xmid[i]) + a.xin[i];

            norm(cfg.norm, a.xmid, w, L.n2g, L.n2b, a.ln2o, a.xhat2, a.rstd2, from, n, D);
            lin(m, a.ln2o, LM.w1, L.b1, a.hpre, from, n, D, F);
            if (swi) {
                lin(m, a.ln2o, LM.wu, L.bu, a.upre, from, n, D, F);
                for (int i = from * F; i < n * F; i++) a.hact[i] = silu(a.hpre[i]) * a.upre[i];
            } else {
                for (int i = from * F; i < n * F; i++) a.hact[i] = gelu(a.hpre[i]);
            }
            lin(m, a.hact, LM.w2, L.b2, a.xout, from, n, F, D);
            if (useDrop) genMask(a.m2, drop, dp, from, n, D);
            for (int i = from * D; i < n * D; i++) a.xout[i] = (useDrop ? a.xout[i] * a.m2[i] : a.xout[i]) + a.xmid[i];
            cur = a.xout;
        }
        norm(cfg.norm, cur, w, lay.nfg, lay.nfb, c.lnfo, c.xhatF, c.rstdF, from, n, D);

        int first = tgt != null ? 0 : n - 1;
        int wsb = qw != null ? qw.sb(ml.wte) : 0;
        double loss = 0;
        for (int t = first; t < n; t++) {
            int pb = t * V;
            final int tt = t;
            parCols(V, V * D, (v0, v1) -> {
                for (int v = v0; v < v1; v++) {
                    double dot = 0;
                    if (qw != null) {
                        dot = qDot(qw, ml.wte, wsb, v * D, D, c.lnfo, tt * D);
                    } else {
                        int eb = lay.wte + v * D;
                        for (int d = 0; d < D; d++) dot += c.lnfo[tt * D + d] * w[eb + d];
                    }
                    c.probs[pb + v] = dot;
                }
                return 0;
            });
            double max = Double.NEGATIVE_INFINITY;
            for (int v = 0; v < V; v++) if (c.probs[pb + v] > max) max = c.probs[pb + v];
            double sum = 0;
            for (int v = 0; v < V; v++) { double e = Math.exp(c.probs[pb + v] - max); c.probs[pb + v] = e; sum += e; }
            for (int v = 0; v < V; v++) c.probs[pb + v] /= sum;
            if (tgt != null) loss -= Math.log(Math.max(c.probs[pb + tgt[t]], 1e-300));
        }
        return tgt != null ? loss / n : 0;
    }

    // =============================================================== backward
    /** Must follow forward(...from=0, with targets, same dp/masks) on the same cache. Accumulates into g. */
    static void backward(Model m, Cache c, int[] tok, int[] tgt, int n, double[] g, double dp) {
        Config cfg = m.cfg; Layout lay = m.lay; double[] w = m.w;
        int D = cfg.dim, F = cfg.ff(), H = cfg.heads, T = cfg.ctx, V = cfg.vocab;
        boolean rope = cfg.pos == POS_ROPE, swi = cfg.act == ACT_SWIGLU, drop = dp > 0;

        Arrays.fill(c.dlnf, 0, n * D, 0);
        for (int t = 0; t < n; t++) {
            int pb = t * V;
            for (int v = 0; v < V; v++) {
                double dl = c.probs[pb + v];
                if (v == tgt[t]) dl -= 1;
                dl /= n;
                int eb = lay.wte + v * D;
                for (int d = 0; d < D; d++) {
                    g[eb + d] += dl * c.lnfo[t * D + d];
                    c.dlnf[t * D + d] += dl * w[eb + d];
                }
            }
        }
        Arrays.fill(c.dx, 0, n * D, 0);
        normBack(cfg.norm, c.xhatF, c.rstdF, w, g, lay.nfg, lay.nfb, c.dlnf, c.dx, n, D);

        for (int l = cfg.layers - 1; l >= 0; l--) {
            LayerOff L = lay.layer[l];
            LayerCache a = c.lc[l];

            double[] dffo = c.dx;
            if (drop) {
                for (int i = 0; i < n * D; i++) c.dtmp[i] = c.dx[i] * a.m2[i];
                dffo = c.dtmp;
            }
            Arrays.fill(c.dhact, 0, n * F, 0);
            linearBack(a.hact, dffo, c.dhact, w, g, L.w2, L.b2, n, F, D);
            Arrays.fill(c.dln2, 0, n * D, 0);
            if (swi) {
                for (int i = 0; i < n * F; i++) {
                    double x = a.hpre[i], u = a.upre[i], dh = c.dhact[i];
                    c.dhpre[i] = dh * u * siluGrad(x);
                    c.dhup[i] = dh * silu(x);
                }
                linearBack(a.ln2o, c.dhpre, c.dln2, w, g, L.w1, L.b1, n, D, F);
                linearBack(a.ln2o, c.dhup, c.dln2, w, g, L.wu, L.bu, n, D, F);
            } else {
                for (int i = 0; i < n * F; i++) c.dhpre[i] = c.dhact[i] * geluGrad(a.hpre[i]);
                linearBack(a.ln2o, c.dhpre, c.dln2, w, g, L.w1, L.b1, n, D, F);
            }
            normBack(cfg.norm, a.xhat2, a.rstd2, w, g, L.n2g, L.n2b, c.dln2, c.dx, n, D); // dx -> grad wrt xmid

            double[] dproj = c.dx;
            if (drop) {
                for (int i = 0; i < n * D; i++) c.dtmp[i] = c.dx[i] * a.m1[i];
                dproj = c.dtmp;
            }
            Arrays.fill(c.datt, 0, n * D, 0);
            linearBack(a.att, dproj, c.datt, w, g, L.wo, L.bo, n, D, D);
            Arrays.fill(c.dq, 0, n * D, 0);
            Arrays.fill(c.dk, 0, n * D, 0);
            Arrays.fill(c.dv, 0, n * D, 0);
            attnBack(a, c.datt, c.dq, c.dk, c.dv, n, T, D, H);
            if (rope) {
                ropeBack(c.dq, c.ropeCos, c.ropeSin, n, D, H);
                ropeBack(c.dk, c.ropeCos, c.ropeSin, n, D, H);
            }
            Arrays.fill(c.dln1, 0, n * D, 0);
            linearBack(a.ln1o, c.dq, c.dln1, w, g, L.wq, L.bq, n, D, D);
            linearBack(a.ln1o, c.dk, c.dln1, w, g, L.wk, L.bk, n, D, D);
            linearBack(a.ln1o, c.dv, c.dln1, w, g, L.wv, L.bv, n, D, D);
            normBack(cfg.norm, a.xhat1, a.rstd1, w, g, L.n1g, L.n1b, c.dln1, c.dx, n, D); // dx -> grad wrt xin
        }
        for (int t = 0; t < n; t++) {
            int e = lay.wte + tok[t] * D;
            for (int d = 0; d < D; d++) g[e + d] += c.dx[t * D + d];
            if (lay.wpe >= 0) for (int d = 0; d < D; d++) g[lay.wpe + t * D + d] += c.dx[t * D + d];
        }
    }

    // ================================================================ corpus
    /** Random access to a token stream; file-backed variants never hold the whole corpus in memory. */
    static abstract class Corpus implements Closeable {
        abstract long length();

        abstract void read(long start, int len, int[] dst) throws IOException;

        @Override public void close() throws IOException { }

        static Corpus open(Path file, boolean tokens32, boolean stream) throws IOException {
            if (stream) return new FileCorpus(file, tokens32);
            byte[] all = Files.readAllBytes(file);
            if (!tokens32) return new MemBytes(all);
            java.nio.IntBuffer ib = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
            int[] t = new int[ib.remaining()];
            ib.get(t);
            return new MemTokens(t);
        }
    }

    static final class MemBytes extends Corpus {
        final byte[] d;
        MemBytes(byte[] d) { this.d = d; }
        long length() { return d.length; }
        void read(long s, int len, int[] dst) { for (int i = 0; i < len; i++) dst[i] = d[(int) s + i] & 0xFF; }
    }

    static final class MemTokens extends Corpus {
        final int[] d;
        MemTokens(int[] d) { this.d = d; }
        long length() { return d.length; }
        void read(long s, int len, int[] dst) { System.arraycopy(d, (int) s, dst, 0, len); }
    }

    static final class FileCorpus extends Corpus {
        final RandomAccessFile f;
        final boolean tok32;
        final long len;

        FileCorpus(Path p, boolean tok32) throws IOException {
            f = new RandomAccessFile(p.toFile(), "r");
            this.tok32 = tok32;
            len = f.length() / (tok32 ? 4 : 1);
        }

        long length() { return len; }

        void read(long s, int n, int[] dst) throws IOException {
            if (!tok32) {
                byte[] b = new byte[n];
                f.seek(s);
                f.readFully(b);
                for (int i = 0; i < n; i++) dst[i] = b[i] & 0xFF;
            } else {
                byte[] b = new byte[n * 4];
                f.seek(s * 4);
                f.readFully(b);
                ByteBuffer bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < n; i++) dst[i] = bb.getInt();
            }
        }

        @Override public void close() throws IOException { f.close(); }
    }

    // ================================================================ training
    static final class TrainParams {
        String data, out, init, resume;
        int dim, heads, layers, ctx, vocab, specials, norm, act, pos, batch, accum, steps, until, eval, threads;
        double lr, wd, clip, dropout;
        long seed;
        boolean stream, quiet;
        int prec, group;

        static TrainParams from(Map<String, String> a) {
            TrainParams p = new TrainParams();
            p.data = need(a, "data"); p.out = need(a, "out");
            p.init = a.get("init"); p.resume = a.get("resume");
            p.dim = geti(a, "dim", 64); p.heads = geti(a, "heads", 4); p.layers = geti(a, "layers", 2);
            p.ctx = geti(a, "ctx", 64); p.vocab = geti(a, "vocab", 256); p.specials = geti(a, "specials", 0);
            p.norm = "ln".equals(a.getOrDefault("norm", "rms")) ? NORM_LN : NORM_RMS;
            p.act = "gelu".equals(a.getOrDefault("act", "swiglu")) ? ACT_GELU : ACT_SWIGLU;
            p.pos = "learned".equals(a.getOrDefault("pos", "rope")) ? POS_LEARNED : POS_ROPE;
            p.batch = geti(a, "batch", 8); p.accum = geti(a, "accum", 1);
            p.steps = geti(a, "steps", 2000); p.until = geti(a, "until", p.steps);
            p.eval = geti(a, "eval", 100);
            p.threads = geti(a, "threads", Runtime.getRuntime().availableProcessors());
            p.lr = getd(a, "lr", 3e-3); p.wd = getd(a, "wd", 0.01); p.clip = getd(a, "clip", 1.0);
            p.dropout = getd(a, "dropout", 0.0);
            p.seed = Long.parseLong(a.getOrDefault("seed", "1"));
            p.prec = parsePrec(a.getOrDefault("precision", "fp64")); p.group = geti(a, "group", 64);
            p.stream = geti(a, "stream", 0) != 0; p.quiet = geti(a, "quiet", 0) != 0;
            return p;
        }
    }

    // ----- emulated reduced precision (Java has no fp16/bf16 arithmetic; values are rounded, math stays double)
    static final int PREC_FP64 = 0, PREC_FP32 = 1, PREC_BF16 = 2, PREC_FP16 = 3, PREC_TERN = 4;

    static double roundTo(double x, int prec) {
        switch (prec) {
            case PREC_FP32: return (float) x;
            case PREC_BF16: {
                float f = (float) x;
                if (Float.isNaN(f) || Float.isInfinite(f)) return f;
                int b = Float.floatToRawIntBits(f);
                b += 0x7FFF + ((b >>> 16) & 1); // round to nearest even
                return Float.intBitsToFloat(b & 0xFFFF0000);
            }
            case PREC_FP16: return Float.float16ToFloat(Float.floatToFloat16((float) x));
            default: return x;
        }
    }

    static int parsePrec(String s) {
        switch (s) {
            case "fp64": return PREC_FP64;
            case "fp32": return PREC_FP32;
            case "bf16": return PREC_BF16;
            case "fp16": return PREC_FP16;
            case "ternary": return PREC_TERN;
            default: throw new IllegalArgumentException("precision must be fp64|fp32|bf16|fp16|ternary");
        }
    }

    static Model trainCore(TrainParams P) throws Exception {
        Path dataPath = Paths.get(P.data);
        Model model;
        Bpe bpe;
        AdamW opt;
        if (P.resume != null) {
            Loaded L = loadCkpt(P.resume);
            if (L.opt == null) throw new IOException("Checkpoint has no optimizer state; use init= instead");
            model = L.model; bpe = L.bpe; opt = L.opt;
            say(P, "Resumed from " + P.resume + " at step " + opt.step);
        } else if (P.init != null) {
            Loaded L = loadCkpt(P.init);
            model = L.model; bpe = L.bpe; opt = new AdamW(model.lay, P.wd);
            say(P, "Loaded weights from " + P.init);
        } else {
            bpe = (P.vocab > 256 || P.specials > 0) ? Bpe.train(Bpe.wordFreq(dataPath, P.specials), P.vocab, P.specials) : new Bpe(new int[0][]);
            Config cfg = new Config(P.dim, P.heads, P.layers, P.ctx, bpe.vocabSize(), P.norm, P.act, P.pos);
            model = new Model(cfg, P.seed);
            opt = new AdamW(model.lay, P.wd);
            if (P.vocab > 256 || P.specials > 0) say(P, "BPE trained: " + bpe.merges.length + " merges + " + bpe.nSpecial + " special tokens, vocab " + bpe.vocabSize());
        }
        Config cfg = model.cfg;
        int T = cfg.ctx;

        Corpus corpus;
        if (bpe.merges.length == 0 && bpe.nSpecial == 0) {
            corpus = Corpus.open(dataPath, false, P.stream);
        } else {
            Path tokenFile = Paths.get(P.out + ".tokens");
            try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tokenFile), 1 << 20))) {
                Bpe.forEachChunk(dataPath, chunk -> {
                    try {
                        int[] ids = bpe.encode(chunk);
                        ByteBuffer bb = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                        for (int id : ids) bb.putInt(id);
                        o.write(bb.array());
                    } catch (IOException e) { throw new UncheckedIOException(e); }
                });
            }
            corpus = Corpus.open(tokenFile, true, P.stream);
        }
        try {
            long N = corpus.length();
            if (N < T + 2) throw new IllegalArgumentException("Data too small for ctx=" + T + " (" + N + " tokens)");
            long split = N > 20L * (T + 1) ? (long) (N * 0.9) : N;
            long vStart = split < N ? split : 0, vEnd = N;
            say(P, String.format("params=%d dim=%d heads=%d layers=%d ctx=%d vocab=%d norm=%s act=%s pos=%s | %d tokens (train %d, val %d) | %s",
                    model.numParams(), cfg.dim, cfg.heads, cfg.layers, T, cfg.vocab,
                    cfg.norm == NORM_LN ? "ln" : "rms", cfg.act == ACT_GELU ? "gelu" : "swiglu",
                    cfg.pos == POS_LEARNED ? "learned" : "rope", N, split, vEnd - vStart,
                    P.stream ? "streaming" : "in-memory"));

            int total = P.batch * P.accum;
            int W = Math.max(1, Math.min(P.threads, total));
            Cache[] caches = new Cache[W];
            double[][] grads = new double[W][model.numParams()];
            for (int i = 0; i < W; i++) caches[i] = new Cache(cfg);
            double[] sum = new double[model.numParams()];
            Cache vcache = new Cache(cfg);
            // master weights stay double; forward/backward use a copy rounded to the compute precision
            final Model cm = P.prec == PREC_FP64 ? model : new Model(cfg, model.lay, new double[model.numParams()]);
            double lossScale = 32768;
            int goodSteps = 0, skipped = 0;
            ExecutorService pool = Executors.newFixedThreadPool(W);
            long t0 = System.nanoTime();
            int startStep = opt.step;
            double ema = Double.NaN;
            try {
                for (int step = opt.step; step < P.until; step++) {
                    if (P.prec == PREC_TERN) ternarize(model.w, cm.w, model.lay, P.group); // straight-through: grads go to the master
                    else if (cm != model) for (int i = 0; i < cm.w.length; i++) cm.w[i] = roundTo(model.w[i], P.prec);
                    Rng sr = new Rng(P.seed * 0x2545F4914F6CDD1DL + step);
                    int[][] bufs = new int[total][T + 1];
                    for (int i = 0; i < total; i++) corpus.read(sr.nextLongBound(split - T - 1), T + 1, bufs[i]);
                    final int stepF = step;
                    List<Callable<Double>> jobs = new ArrayList<>();
                    for (int k = 0; k < W; k++) {
                        final int id = k;
                        jobs.add(() -> {
                            Arrays.fill(grads[id], 0);
                            int[] tok = new int[T], tgt = new int[T];
                            double ls = 0;
                            for (int i = id; i < total; i += W) {
                                System.arraycopy(bufs[i], 0, tok, 0, T);
                                System.arraycopy(bufs[i], 1, tgt, 0, T);
                                Rng dr = P.dropout > 0 ? new Rng(P.seed * 0x9E3779B1L + stepF * 1000003L + i) : null;
                                ls += forward(cm, caches[id], tok, tgt, 0, T, dr, P.dropout);
                                backward(cm, caches[id], tok, tgt, T, grads[id], P.dropout);
                            }
                            return ls;
                        });
                    }
                    double loss = 0;
                    for (Future<Double> f : pool.invokeAll(jobs)) loss += f.get();
                    loss /= total;
                    double norm = 0;
                    boolean overflow = false;
                    for (int i = 0; i < sum.length; i++) {
                        double s = 0;
                        for (int k = 0; k < W; k++) s += grads[k][i];
                        s /= total;
                        if (P.prec == PREC_FP16) {
                            // fp16 gradients: scaled up to avoid underflow, rounded, checked for overflow, unscaled
                            double g16 = roundTo(s * lossScale, PREC_FP16);
                            if (Double.isInfinite(g16) || Double.isNaN(g16)) { overflow = true; break; }
                            s = g16 / lossScale;
                        } else if (P.prec != PREC_FP64) {
                            s = roundTo(s, P.prec);
                        }
                        sum[i] = s;
                        norm += s * s;
                    }
                    if (overflow) {
                        lossScale = Math.max(1, lossScale / 2);
                        goodSteps = 0;
                        skipped++;
                        step--; // retry the same step (same batch) with a smaller scale
                        if (skipped > 1000) throw new IllegalStateException("fp16 gradients keep overflowing");
                        continue;
                    }
                    if (P.prec == PREC_FP16 && ++goodSteps >= 200 && lossScale < 65536) { lossScale *= 2; goodSteps = 0; }
                    norm = Math.sqrt(norm);
                    if (norm > P.clip) for (int i = 0; i < sum.length; i++) sum[i] *= P.clip / norm;
                    opt.update(model.w, sum, lrAt(step, P.steps, P.lr));

                    ema = Double.isNaN(ema) ? loss : 0.98 * ema + 0.02 * loss;
                    boolean last = step + 1 == P.until;
                    if ((step + 1) % P.eval == 0 || last) {
                        double vl = evalLoss(cm, vcache, corpus, vStart, vEnd, P.seed + 777, 16);
                        double secs = (System.nanoTime() - t0) / 1e9;
                        say(P, String.format("step %5d/%d | train %.4f | val %.4f (ppl %.2f) | |g| %.3f | %.0f tok/s | %.1fs",
                                step + 1, P.steps, ema, vl, Math.exp(vl), norm,
                                (double) (step + 1 - startStep) * total * T / secs, secs));
                        saveCkpt(P.out, model, bpe, opt);
                    }
                }
            } finally {
                pool.shutdown();
            }
            saveCkpt(P.out, model, bpe, opt);
            say(P, "Saved " + P.out);
        } finally {
            corpus.close();
        }
        return model;
    }

    static void say(TrainParams P, String s) { if (!P.quiet) System.out.println(s); }

    static double evalLoss(Model m, Cache c, Corpus corpus, long from, long to, long seed, int windows) throws IOException {
        int T = m.cfg.ctx;
        long span = to - from - T - 1;
        if (span <= 0) return Double.NaN;
        Rng r = new Rng(seed);
        int[] buf = new int[T + 1], tok = new int[T], tgt = new int[T];
        double sum = 0;
        for (int k = 0; k < windows; k++) {
            corpus.read(from + r.nextLongBound(span), T + 1, buf);
            System.arraycopy(buf, 0, tok, 0, T);
            System.arraycopy(buf, 1, tgt, 0, T);
            sum += forward(m, c, tok, tgt, 0, T, null, 0);
        }
        return sum / windows;
    }

    // ============================================================== generation
    /** Loads model=...; quantized files stay byte-resident unless resident=0; quant=8|ternary converts an fp64 checkpoint in memory. */
    static Loaded loadModel(Map<String, String> a) throws IOException {
        inferThreads = Math.max(1, geti(a, "threads", Runtime.getRuntime().availableProcessors()));
        Loaded L = loadCkpt(need(a, "model"), geti(a, "resident", 1) != 0);
        String q = a.get("quant");
        if (q != null && L.model.qw == null) {
            boolean tern = "ternary".equals(q);
            if (!tern && !"8".equals(q)) throw new IllegalArgumentException("quant must be 8 or ternary");
            L.model = residentFrom(L.model, geti(a, "group", 64), tern);
        }
        return L;
    }

    static long weightBytes(Model m) {
        long b = m.w.length * 8L;
        if (m.qw != null) b += m.qw.q.length + m.qw.scale.length * 4L;
        return b;
    }

    /**
     * Held-out evaluation over a whole file: every token is scored exactly once, in order, with sliding windows
     * (stride = number of new tokens scored per window; each scored token sees at least ctx-stride tokens of context,
     * except at the very start of the file). Deterministic; unlike `eval` there is no random window sampling.
     */
    static double[] slidingLoss(Model m, Cache c, int[] x, int stride) {
        int T = m.cfg.ctx, V = m.cfg.vocab, N = x.length;
        if (N < T + 2) throw new IllegalArgumentException("need at least ctx+2 tokens (" + (T + 2) + "), have " + N);
        stride = Math.max(1, Math.min(stride, T));
        int[] tok = new int[T], tgt = new int[T]; // targets make forward() fill probs for every position
        double sum = 0;
        long scored = 0;
        int next = 1;
        while (next < N) {
            int s = Math.max(0, Math.min(next - 1 - (T - stride), N - 1 - T));
            System.arraycopy(x, s, tok, 0, T);
            forward(m, c, tok, tgt, 0, T, null, 0);
            for (int g = Math.max(next, s + 1); g <= s + T; g++) {
                sum -= Math.log(Math.max(1e-300, c.probs[(g - 1 - s) * V + x[g]]));
                scored++;
            }
            next = s + T + 1;
        }
        return new double[]{sum / scored, scored};
    }

    static void heldoutCmd(Map<String, String> a) throws Exception {
        Loaded L = loadModel(a);
        byte[] raw = Files.readAllBytes(Paths.get(need(a, "data")));
        int[] x = L.bpe.encode(raw);
        int stride = geti(a, "stride", Math.max(1, L.model.cfg.ctx / 2));
        long t0 = System.nanoTime();
        double[] r = slidingLoss(L.model, new Cache(L.model.cfg), x, stride);
        System.out.printf("held-out: %d tokens scored (every token once, stride %d) | loss %.4f | perplexity %.2f | bits/byte %.3f | %.1f s | weights %,d bytes%n",
                (long) r[1], stride, r[0], Math.exp(r[0]), r[0] * x.length / (Math.log(2) * raw.length), (System.nanoTime() - t0) / 1e9, weightBytes(L.model));
    }

    /** Held-out style evaluation: mean cross-entropy over random windows of a text file. */
    static void evalCmd(Map<String, String> a) throws Exception {
        Loaded L = loadModel(a);
        byte[] raw = Files.readAllBytes(Paths.get(need(a, "data")));
        Corpus cp = new MemTokens(L.bpe.encode(raw));
        int windows = geti(a, "windows", 50);
        double loss = evalLoss(L.model, new Cache(L.model.cfg), cp, 0, cp.length(), Long.parseLong(a.getOrDefault("seed", "7")), windows);
        System.out.printf("tokens %d | windows %d | loss %.4f | perplexity %.2f | bits/byte %.3f | weights in memory %,d bytes%n",
                cp.length(), windows, loss, Math.exp(loss), loss * cp.length() / (Math.log(2) * raw.length), weightBytes(L.model));
    }

    /** Generation speed and weight memory for double vs int8 vs ternary resident weights (random weights; speed only). */
    static void benchCmd(Map<String, String> a) {
        Config cfg = new Config(geti(a, "dim", 512), geti(a, "heads", 8), geti(a, "layers", 8), geti(a, "ctx", 128), 256,
                NORM_RMS, ACT_SWIGLU, POS_ROPE);
        Model fp = new Model(cfg, 1);
        Bpe plain = new Bpe(new int[0][]);
        int len = geti(a, "len", 100);
        System.out.printf("dim %d layers %d ctx %d: %,d parameters%n", cfg.dim, cfg.layers, cfg.ctx, fp.numParams());
        String[] names = {"double", "int8 g64", "int8 row-scale", "ternary g64", "ternary row-scale"};
        int cores = Runtime.getRuntime().availableProcessors();
        int[] sweep = a.containsKey("sweep") ? Arrays.stream(a.get("sweep").split(",")).mapToInt(Integer::parseInt).toArray() : new int[]{1, 2, 4, 8, cores};
        System.out.printf("%-18s", "tok/s by threads");
        for (int T : sweep) System.out.printf("%10s", "T=" + T);
        System.out.println();
        for (int k = 0; k < 5; k++) {
            Model m = k == 0 ? fp : residentFrom(fp, k % 2 == 1 ? 64 : 0, k >= 3);
            System.out.printf("%-18s", names[k]);
            for (int T : sweep) {
                inferThreads = T;
                generateBytes(m, plain, "hello".getBytes(), 8, 0.8, 40, new Rng(1));
                double best = 0;
                for (int rep = 0; rep < 3; rep++) {
                    long t0 = System.nanoTime();
                    generateBytes(m, plain, "hello".getBytes(), len, 0.8, 40, new Rng(1));
                    best = Math.max(best, len / ((System.nanoTime() - t0) / 1e9));
                }
                System.out.printf("%10.1f", best);
            }
            System.out.printf("   weights %,d bytes%n", weightBytes(m));
        }
        inferThreads = 1;
    }

    static void generate(Map<String, String> a) throws Exception {
        Loaded L = loadModel(a);
        String prompt = a.containsKey("promptfile")
                ? new String(Files.readAllBytes(Paths.get(a.get("promptfile"))), StandardCharsets.UTF_8)
                : a.getOrDefault("prompt", "\n");
        int len = geti(a, "len", 300), topk = geti(a, "topk", 40);
        double temp = getd(a, "temp", 0.8);
        Rng rng = new Rng(Long.parseLong(a.getOrDefault("seed", "1")));
        byte[] out = generateBytes(L.model, L.bpe, prompt.getBytes(StandardCharsets.UTF_8), len, temp, topk, rng,
                geti(a, "trim", 0) != 0);
        System.out.write(out);
        System.out.println();
        System.out.flush();
    }

    /**
     * Slides the cache window left by `drop` rows: rows [drop,n) become [0,n-drop). V is position-free; K is stored
     * post-RoPE, so each kept key is rotated by -drop positions (rotations compose), which makes it exactly the key
     * it would have at its new position given the same hidden state.
     */
    static void trimCache(Model m, Cache c, int drop, int n) {
        int D = m.cfg.dim, H = m.cfg.heads, hd = D / H, half = hd / 2, rows = n - drop;
        for (LayerCache a : c.lc) {
            System.arraycopy(a.k, drop * D, a.k, 0, rows * D);
            System.arraycopy(a.v, drop * D, a.v, 0, rows * D);
            for (int t = 0; t < rows; t++)
                for (int h = 0; h < H; h++)
                    for (int i = 0; i < half; i++) {
                        int i1 = t * D + h * hd + i, i2 = i1 + half;
                        double co = c.ropeCos[drop * half + i], si = c.ropeSin[drop * half + i];
                        double x = a.k[i1], y = a.k[i2];
                        a.k[i1] = x * co + y * si;
                        a.k[i2] = -x * si + y * co;
                    }
        }
    }

    static byte[] generateBytes(Model m, Bpe bpe, byte[] prompt, int len, double temp, int topk, Rng rng) {
        return generateBytes(m, bpe, prompt, len, temp, topk, rng, false);
    }

    /**
     * KV-cached generation. When the window fills, it keeps the most recent half. Default: re-prefill it (exact).
     * With trim (RoPE only): slide the cache instead (no re-prefill, ~2x less work, but kept rows retain the context
     * they were computed with, so it is an approximation for models with more than one layer).
     */
    static byte[] generateBytes(Model m, Bpe bpe, byte[] prompt, int len, double temp, int topk, Rng rng, boolean trim) {
        return generateBytes(m, bpe, prompt, len, temp, topk, rng, trim, false);
    }

    /** stopAtEos: end generation (without emitting it) when the model produces the end-of-text special token. */
    static byte[] generateBytes(Model m, Bpe bpe, byte[] prompt, int len, double temp, int topk, Rng rng, boolean trim, boolean stopAtEos) {
        trim = trim && m.cfg.pos == POS_ROPE;
        int T = m.cfg.ctx, V = m.cfg.vocab;
        Cache c = new Cache(m.cfg);
        int[] ids = bpe.encode(prompt);
        if (ids.length == 0) ids = new int[]{'\n'};
        List<Integer> all = new ArrayList<>();
        for (int id : ids) all.add(id);
        int[] win = new int[T];
        int n = Math.min(T, ids.length);
        System.arraycopy(ids, ids.length - n, win, 0, n);
        int valid = 0;
        for (int i = 0; i < len; i++) {
            forward(m, c, win, null, valid, n, null, 0);
            valid = n;
            int next = sample(c.probs, (n - 1) * V, V, temp, topk, rng);
            if (stopAtEos && next == bpe.eosId()) break;
            all.add(next);
            if (n < T) {
                win[n++] = next;
            } else {
                int keep = Math.max(1, T / 2);
                if (trim) trimCache(m, c, T - (keep - 1), T);
                System.arraycopy(win, T - (keep - 1), win, 0, keep - 1);
                win[keep - 1] = next;
                n = keep;
                valid = trim ? keep - 1 : 0;
            }
        }
        int[] arr = new int[all.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = all.get(i);
        return bpe.decode(arr, arr.length);
    }

    static int sample(double[] probs, int off, int V, double temp, int topk, Rng rng) {
        double[] p = new double[V];
        if (temp <= 1e-6) {
            int best = 0;
            for (int i = 1; i < V; i++) if (probs[off + i] > probs[off + best]) best = i;
            return best;
        }
        for (int i = 0; i < V; i++) p[i] = Math.pow(Math.max(probs[off + i], 1e-300), 1.0 / temp);
        if (topk > 0 && topk < V) {
            double[] sorted = p.clone();
            Arrays.sort(sorted);
            double thr = sorted[V - topk];
            for (int i = 0; i < V; i++) if (p[i] < thr) p[i] = 0;
        }
        double sum = 0;
        for (double x : p) sum += x;
        double r = rng.nextDouble() * sum, acc = 0;
        for (int i = 0; i < V; i++) {
            acc += p[i];
            if (r < acc) return i;
        }
        return V - 1;
    }

    // ================================================================ selftest
    static int fails = 0, passes = 0;

    /** Verifies pieces() against java.util.regex running the GPT-2 pattern (an independent oracle). */
    static void pretokTests() {
        java.util.regex.Pattern gpt2 = java.util.regex.Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
            java.util.regex.Pattern.UNICODE_CHARACTER_CLASS);
        java.util.regex.Matcher letter = java.util.regex.Pattern.compile("\\p{L}", java.util.regex.Pattern.UNICODE_CHARACTER_CLASS).matcher("");
        java.util.regex.Matcher number = java.util.regex.Pattern.compile("\\p{N}", java.util.regex.Pattern.UNICODE_CHARACTER_CLASS).matcher("");
        java.util.regex.Matcher space = java.util.regex.Pattern.compile("\\s", java.util.regex.Pattern.UNICODE_CHARACTER_CLASS).matcher("");
        int bad = 0, checked = 0, firstBad = -1;
        for (int cp = 0; cp <= 0x10FFFF; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) continue;
            String s = new String(Character.toChars(cp));
            int want = letter.reset(s).matches() ? Bpe.LETTER : number.reset(s).matches() ? Bpe.DIGIT
                : space.reset(s).matches() ? Bpe.SPACE : Bpe.OTHER;
            byte[] u = s.getBytes(StandardCharsets.UTF_8);
            int got = Bpe.charAt(u, 0, u.length);
            checked++;
            if ((got & 3) != want || (got >> 2) != u.length) { bad++; if (firstBad < 0) firstBad = cp; }
        }
        report("char classes match regex for every code point", bad == 0,
            checked + " checked" + (bad > 0 ? ", " + bad + " differ, first U+" + Integer.toHexString(firstBad) : ""));

        String[] fixed = {"", " ", "  ", "hello world", "Hello,  world!!", "don't we're they've I'm you'll he'd it's",
            "x2y 3.14 abc123def", "a\n\nb", "a \n b", "tab\t\tend", "trailing   ", "  leading", "'", "''s", "it's'", "\n\n\n",
            "naïve café — 日本語 🙂", "snake_case __init__ f(x)=y", "a\u3000\u3000b", "line1\r\nline2\r\n"};
        ArrayList<String> cases = new ArrayList<>(Arrays.asList(fixed));
        String alphabet = "ab Z9'_-.,\n\t sSrelvmdt\u00e9\u65e5\u3000\u2014";
        Rng rng = new Rng(77);
        for (int c = 0; c < 4000; c++) {
            StringBuilder sb = new StringBuilder();
            int len = rng.nextInt(24);
            for (int i = 0; i < len; i++) sb.append(alphabet.charAt(rng.nextInt(alphabet.length())));
            cases.add(sb.toString());
        }
        int diff = 0;
        String firstDiff = null;
        for (String s : cases) {
            ArrayList<String> want = new ArrayList<>();
            java.util.regex.Matcher m = gpt2.matcher(s);
            while (m.find()) want.add(m.group());
            byte[] u = s.getBytes(StandardCharsets.UTF_8);
            ArrayList<String> got = new ArrayList<>();
            Bpe.pieces(u, 0, u.length, (a, e) -> got.add(new String(u, a, e - a, StandardCharsets.UTF_8)));
            if (!want.equals(got)) { diff++; if (firstDiff == null) firstDiff = "[" + s + "] want " + want + " got " + got; }
        }
        report("pretokenizer == GPT-2 regex", diff == 0, cases.size() + " strings" + (diff > 0 ? ", " + diff + " differ, e.g. " + firstDiff : ""));
    }

    static void report(String name, boolean ok, String info) {
        if (ok) passes++; else fails++;
        System.out.printf("  [%s] %s  %s%n", ok ? "PASS" : "FAIL", name, info);
    }

    static double[] rnd(Rng r, int n) {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = r.nextGaussian();
        return a;
    }

    static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    /** Max relative error between analytic gradients and central finite differences. */
    static double gradOp(double[][] in, double[][] ana, DoubleSupplier loss) {
        double worst = 0, h = 1e-6;
        Rng r = new Rng(11);
        for (int k = 0; k < in.length; k++) {
            int cnt = Math.min(in[k].length, 60);
            for (int j = 0; j < cnt; j++) {
                int i = in[k].length <= 60 ? j : r.nextInt(in[k].length);
                double o = in[k][i];
                in[k][i] = o + h; double lp = loss.getAsDouble();
                in[k][i] = o - h; double lm = loss.getAsDouble();
                in[k][i] = o;
                double num = (lp - lm) / (2 * h);
                worst = Math.max(worst, Math.abs(num - ana[k][i]) / Math.max(1e-8, Math.abs(num) + Math.abs(ana[k][i])));
            }
        }
        return worst;
    }

    static void opTests() {
        final double TOL = 1e-6;
        Rng r = new Rng(1);
        {
            int n = 5, in = 7, out = 6; // in=7 exercises the 4-way unrolled loop and its remainder
            double[] x = rnd(r, n * in), w = rnd(r, in * out + out), dy = rnd(r, n * out), y = new double[n * out];
            double[] dx = new double[n * in], gw = new double[w.length];
            linearBack(x, dy, dx, w, gw, 0, in * out, n, in, out);
            double e = gradOp(new double[][]{x, w}, new double[][]{dx, gw}, () -> { linear(x, w, 0, in * out, y, 0, n, in, out); return dot(y, dy); });
            report("linear backward", e < TOL, String.format("rel err %.2e", e));
            double[] y2 = new double[n * out];
            for (int t = 0; t < n; t++) for (int o = 0; o < out; o++) {
                double s = w[in * out + o];
                for (int i = 0; i < in; i++) s += x[t * in + i] * w[i * out + o];
                y2[t * out + o] = s;
            }
            linear(x, w, 0, in * out, y, 0, n, in, out);
            report("linear forward vs naive", maxDiff(y, y2) < 1e-12, String.format("max diff %.2e", maxDiff(y, y2)));
        }
        for (int kind = 0; kind < 2; kind++) {
            int n = 3, D = 6;
            final int kk = kind;
            double[] x = rnd(r, n * D), w = rnd(r, 2 * D), dy = rnd(r, n * D), y = new double[n * D];
            double[] xhat = new double[n * D], rs = new double[n];
            double[] dx = new double[n * D], gw = new double[w.length];
            norm(kk, x, w, 0, D, y, xhat, rs, 0, n, D);
            normBack(kk, xhat, rs, w, gw, 0, D, dy, dx, n, D);
            double e = gradOp(new double[][]{x, w}, new double[][]{dx, gw}, () -> { norm(kk, x, w, 0, D, y, xhat, rs, 0, n, D); return dot(y, dy); });
            report(kind == NORM_LN ? "LayerNorm backward" : "RMSNorm backward", e < TOL, String.format("rel err %.2e", e));
        }
        {
            double[] x = rnd(r, 30), dy = rnd(r, 30), dx = new double[30], y = new double[30];
            for (int i = 0; i < 30; i++) dx[i] = dy[i] * geluGrad(x[i]);
            double e = gradOp(new double[][]{x}, new double[][]{dx}, () -> { for (int i = 0; i < 30; i++) y[i] = gelu(x[i]); return dot(y, dy); });
            report("GELU backward", e < TOL, String.format("rel err %.2e", e));
            double[] da = new double[30], u = rnd(r, 30), du = new double[30];
            for (int i = 0; i < 30; i++) { da[i] = dy[i] * u[i] * siluGrad(x[i]); du[i] = dy[i] * silu(x[i]); }
            e = gradOp(new double[][]{x, u}, new double[][]{da, du}, () -> { for (int i = 0; i < 30; i++) y[i] = silu(x[i]) * u[i]; return dot(y, dy); });
            report("SwiGLU gate backward", e < TOL, String.format("rel err %.2e", e));
        }
        {
            Config cfg = new Config(8, 2, 1, 6, 256, NORM_RMS, ACT_GELU, POS_ROPE);
            Cache c = new Cache(cfg);
            int n = 6, D = 8, H = 2;
            double[] x = rnd(r, n * D), dy = rnd(r, n * D), y = new double[n * D];
            double[] dx = dy.clone();
            ropeBack(dx, c.ropeCos, c.ropeSin, n, D, H);
            double e = gradOp(new double[][]{x}, new double[][]{dx}, () -> { System.arraycopy(x, 0, y, 0, y.length); ropeApply(y, c.ropeCos, c.ropeSin, 0, n, D, H); return dot(y, dy); });
            report("RoPE backward", e < TOL, String.format("rel err %.2e", e));

            double[] q = rnd(r, D), k = rnd(r, D);
            double worst = 0;
            double ref3 = ropeDot(c, q, k, 3, 0, D, H);
            worst = Math.max(worst, Math.abs(ropeDot(c, q, k, 4, 1, D, H) - ref3));
            worst = Math.max(worst, Math.abs(ropeDot(c, q, k, 5, 2, D, H) - ref3));
            double other = ropeDot(c, q, k, 4, 0, D, H); // different offset must differ
            report("RoPE score depends only on relative position", worst < 1e-12 && Math.abs(other - ref3) > 1e-6, String.format("max diff %.2e", worst));
            double[] z = x.clone();
            ropeApply(z, c.ropeCos, c.ropeSin, 0, n, D, H);
            double diff = Math.abs(dot(x, x) - dot(z, z));
            report("RoPE preserves vector norm", diff < 1e-10, String.format("diff %.2e", diff));
        }
        {
            int T = 5, D = 8, H = 2;
            LayerCache a = new LayerCache(T, D, 8, H);
            for (int i = 0; i < T * D; i++) { a.q[i] = r.nextGaussian(); a.k[i] = r.nextGaussian(); a.v[i] = r.nextGaussian(); }
            double[] dy = rnd(r, T * D);
            attnForward(a, 0, T, T, D, H);
            double[] dq = new double[T * D], dk = new double[T * D], dv = new double[T * D];
            attnBack(a, dy, dq, dk, dv, T, T, D, H);
            double e = gradOp(new double[][]{a.q, a.k, a.v}, new double[][]{dq, dk, dv}, () -> { attnForward(a, 0, T, T, D, H); return dot(a.att, dy); });
            report("attention backward", e < TOL, String.format("rel err %.2e", e));
            attnForward(a, 0, T, T, D, H);
            double[] ref = a.att.clone();
            for (int i = 3 * D; i < T * D; i++) { a.k[i] += 1; a.v[i] += 1; }
            attnForward(a, 0, T, T, D, H);
            double leak = 0;
            for (int i = 0; i < 3 * D; i++) leak = Math.max(leak, Math.abs(ref[i] - a.att[i]));
            report("attention is causal", leak == 0, String.format("leak %.1e", leak));
        }
    }

    /** Dot product of q placed at position tq and k at position tk, after RoPE. */
    static double ropeDot(Cache c, double[] q, double[] k, int tq, int tk, int D, int H) {
        int n = Math.max(tq, tk) + 1;
        double[] a = new double[n * D], b = new double[n * D];
        System.arraycopy(q, 0, a, tq * D, D);
        System.arraycopy(k, 0, b, tk * D, D);
        ropeApply(a, c.ropeCos, c.ropeSin, 0, n, D, H);
        ropeApply(b, c.ropeCos, c.ropeSin, 0, n, D, H);
        double s = 0;
        for (int i = 0; i < D; i++) s += a[tq * D + i] * b[tk * D + i];
        return s;
    }

    static Config[] testConfigs() {
        return new Config[]{
                new Config(16, 4, 2, 8, 256, NORM_LN, ACT_GELU, POS_LEARNED),
                new Config(16, 4, 2, 8, 300, NORM_RMS, ACT_SWIGLU, POS_ROPE),
                new Config(16, 2, 2, 8, 256, NORM_LN, ACT_SWIGLU, POS_LEARNED),
                new Config(16, 2, 1, 8, 270, NORM_RMS, ACT_GELU, POS_ROPE),
        };
    }

    static String name(Config c) {
        return (c.norm == NORM_LN ? "ln" : "rms") + "/" + (c.act == ACT_GELU ? "gelu" : "swiglu") + "/"
                + (c.pos == POS_LEARNED ? "learned" : "rope") + "/v" + c.vocab;
    }

    static double modelGradcheck(Config cfg, double dp) {
        Model m = new Model(cfg, 42);
        Rng r = new Rng(5);
        for (int i = 0; i < m.w.length; i++) m.w[i] += 0.05 * r.nextGaussian();
        int n = cfg.ctx;
        int[] tok = new int[n], tgt = new int[n];
        for (int t = 0; t < n; t++) { tok[t] = r.nextInt(cfg.vocab); tgt[t] = r.nextInt(cfg.vocab); }
        Cache c = new Cache(cfg);
        double[] g = new double[m.w.length];
        forward(m, c, tok, tgt, 0, n, new Rng(99), dp);
        backward(m, c, tok, tgt, n, g, dp);
        double worst = 0, h = 1e-5;
        for (int k = 0; k < 300; k++) {
            int i = r.nextInt(m.w.length);
            double o = m.w[i];
            m.w[i] = o + h; double lp = forward(m, c, tok, tgt, 0, n, new Rng(99), dp);
            m.w[i] = o - h; double lm = forward(m, c, tok, tgt, 0, n, new Rng(99), dp);
            m.w[i] = o;
            double num = (lp - lm) / (2 * h);
            worst = Math.max(worst, Math.abs(num - g[i]) / Math.max(1e-8, Math.abs(num) + Math.abs(g[i])));
        }
        return worst;
    }

    static double kvEquivalence(Config cfg) {
        Model m = new Model(cfg, 7);
        Rng r = new Rng(3);
        for (int i = 0; i < m.w.length; i++) m.w[i] += 0.05 * r.nextGaussian();
        int n = cfg.ctx;
        int[] tok = new int[n];
        for (int t = 0; t < n; t++) tok[t] = r.nextInt(cfg.vocab);
        Cache full = new Cache(cfg), inc = new Cache(cfg);
        double worst = 0;
        int V = cfg.vocab, from = 0;
        for (int cut : new int[]{3, 4, 7, n}) {
            forward(m, inc, tok, null, from, cut, null, 0);
            from = cut;
            forward(m, full, tok, null, 0, cut, null, 0);
            for (int v = 0; v < V; v++) worst = Math.max(worst, Math.abs(full.probs[(cut - 1) * V + v] - inc.probs[(cut - 1) * V + v]));
        }
        return worst;
    }

    static byte[] sampleText(Rng r, int chars) {
        String[] words = {"the", "model", "dreams", "of", "neural", "networks", "and", "learns", "to", "predict", "every",
                "byte", "token", "in", "a", "long", "stream", "fraynix", "attention", "transformer", "gradient", "descent", "."};
        StringBuilder sb = new StringBuilder();
        while (sb.length() < chars) sb.append(words[r.nextInt(words.length)]).append(r.nextInt(12) == 0 ? "\n" : " ");
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    static HashMap<String, String> baseArgs(Path dir, String out) {
        HashMap<String, String> a = new HashMap<>();
        a.put("data", dir.resolve("corpus.txt").toString());
        a.put("out", dir.resolve(out).toString());
        a.put("dim", "16"); a.put("heads", "2"); a.put("layers", "2"); a.put("ctx", "16");
        a.put("vocab", "300"); a.put("batch", "4"); a.put("steps", "6"); a.put("eval", "3");
        a.put("seed", "3"); a.put("threads", "2"); a.put("dropout", "0.1"); a.put("quiet", "1");
        return a;
    }

    static double maxDiff(double[] a, double[] b) {
        double d = 0;
        for (int i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
        return d;
    }

    /** Max |p_trimmed - p_reprefilled| for the next-token distribution right after the window slides. */
    static double trimGap(Config cfg, long seed) {
        Model m = new Model(cfg, seed);
        int T = cfg.ctx, keep = T / 2, drop = T - (keep - 1);
        Rng r = new Rng(seed + 1);
        int[] win = new int[T];
        for (int i = 0; i < T; i++) win[i] = (int) (r.nextDouble() * cfg.vocab);
        Cache a = new Cache(cfg);
        forward(m, a, win, null, 0, T, null, 0);
        trimCache(m, a, drop, T);
        int[] w2 = new int[T];
        System.arraycopy(win, drop, w2, 0, keep - 1);
        w2[keep - 1] = (int) (r.nextDouble() * cfg.vocab);
        forward(m, a, w2, null, keep - 1, keep, null, 0);
        Cache b = new Cache(cfg);
        forward(m, b, w2, null, 0, keep, null, 0);
        int V = cfg.vocab, o = (keep - 1) * V;
        double d = 0;
        for (int i = 0; i < V; i++) d = Math.max(d, Math.abs(a.probs[o + i] - b.probs[o + i]));
        return d;
    }

    static void selftest() throws Exception {
        System.out.println("== per-op gradient checks ==");
        opTests();

        System.out.println("== whole-model gradient checks ==");
        for (Config c : testConfigs()) {
            double e = modelGradcheck(c, 0);
            report("gradcheck " + name(c), e < 1e-4, String.format("rel err %.2e", e));
        }
        double e = modelGradcheck(testConfigs()[1], 0.25);
        report("gradcheck with dropout (rms/swiglu/rope)", e < 1e-4, String.format("rel err %.2e", e));
        e = modelGradcheck(testConfigs()[0], 0.25);
        report("gradcheck with dropout (ln/gelu/learned)", e < 1e-4, String.format("rel err %.2e", e));

        System.out.println("== KV cache / generation ==");
        for (Config c : testConfigs()) {
            double d = kvEquivalence(c);
            report("incremental == full forward " + name(c), d < 1e-12, String.format("max prob diff %.2e", d));
        }
        {
            Bpe plain = new Bpe(new int[0][]);
            Config big = new Config(16, 4, 2, 64, 256, NORM_RMS, ACT_SWIGLU, POS_ROPE);
            byte[] g1 = generateBytes(new Model(big, 9), plain, "ab".getBytes(), 40, 0.9, 20, new Rng(4));
            byte[] g2 = generateBytes(new Model(big, 9), plain, "ab".getBytes(), 40, 0.9, 20, new Rng(4));
            report("generation deterministic for a seed", Arrays.equals(g1, g2) && g1.length == 42, "len " + g1.length);
            Config small = new Config(16, 4, 1, 8, 256, NORM_LN, ACT_GELU, POS_LEARNED);
            byte[] g3 = generateBytes(new Model(small, 9), plain, "x".getBytes(), 50, 0.9, 20, new Rng(1));
            report("generation slides past ctx", g3.length == 51, "len " + g3.length);

            Config one = new Config(16, 4, 1, 16, 256, NORM_RMS, ACT_SWIGLU, POS_ROPE);
            double d1 = trimGap(one, 11);
            report("KV trim == re-prefill (1 layer, exact case)", d1 < 1e-12, String.format("max prob diff %.2e", d1));
            Config two = new Config(16, 4, 3, 16, 256, NORM_RMS, ACT_SWIGLU, POS_ROPE);
            double d3 = trimGap(two, 11);
            report("KV trim ~ re-prefill (3 layers, approximation)", d3 < 0.05, String.format("max prob diff %.2e (not exact by design)", d3));
            Config oneLn = new Config(16, 2, 1, 12, 256, NORM_LN, ACT_GELU, POS_ROPE);
            byte[] t1 = generateBytes(new Model(oneLn, 5), plain, "hello".getBytes(), 80, 0, 0, new Rng(1), false);
            byte[] t2 = generateBytes(new Model(oneLn, 5), plain, "hello".getBytes(), 80, 0, 0, new Rng(1), true);
            report("greedy generation identical with trim (1 layer)", Arrays.equals(t1, t2), "len " + t2.length);
        }

        System.out.println("== dropout ==");
        {
            double[] mk = new double[200000];
            genMask(mk, new Rng(1), 0.3, 0, 100, 2000);
            double mean = 0;
            int zeros = 0;
            for (double v : mk) { mean += v; if (v == 0) zeros++; }
            mean /= mk.length;
            report("mask is unbiased (E=1)", Math.abs(mean - 1) < 0.01, String.format("mean %.4f", mean));
            report("mask drop rate", Math.abs(zeros / (double) mk.length - 0.3) < 0.01, String.format("%.4f", zeros / (double) mk.length));
        }

        System.out.println("== BPE tokenizer ==");
        Path dir = Files.createTempDirectory("singlefilegpt-selftest");
        try {
            byte[] text = sampleText(new Rng(8), 30000);
            Files.write(dir.resolve("corpus.txt"), text);
            HashMap<String, Long> freq = Bpe.wordFreq(dir.resolve("corpus.txt"));
            Bpe bpe = Bpe.train(freq, 320);
            Bpe bpe2 = Bpe.train(freq, 320);
            report("trains requested merges", bpe.merges.length == 64, bpe.merges.length + " merges");
            report("training is deterministic", Arrays.deepEquals(bpe.merges, bpe2.merges), "");
            report("matches naive reference trainer", Arrays.deepEquals(bpe.merges, naiveBpe(freq, 320)), "");
            int[] ids = bpe.encode(text);
            report("roundtrip on corpus", Arrays.equals(bpe.decode(ids, ids.length), text), "");
            report("compresses text", ids.length < text.length * 0.7, String.format("%.1f%% of bytes", 100.0 * ids.length / text.length));
            Rng rr = new Rng(2);
            byte[] junk = new byte[5000];
            for (int i = 0; i < junk.length; i++) junk[i] = (byte) rr.nextInt(256);
            int[] jid = bpe.encode(junk);
            report("roundtrip on random bytes", Arrays.equals(bpe.decode(jid, jid.length), junk), "");
            report("empty input", bpe.encode(new byte[0]).length == 0, "");
            byte[] utf = "naïve café — 日本語 🙂".getBytes(StandardCharsets.UTF_8);
            int[] uid = bpe.encode(utf);
            report("roundtrip on UTF-8", Arrays.equals(bpe.decode(uid, uid.length), utf), "");
            pretokTests();

            System.out.println("== training: checkpoint, resume, accumulation, streaming ==");
            Model ma = trainCore(TrainParams.from(baseArgs(dir, "a.bin")));
            Loaded la = loadCkpt(dir.resolve("a.bin").toString());
            report("checkpoint roundtrip (weights)", maxDiff(ma.w, la.model.w) == 0, "");
            report("checkpoint carries tokenizer + optimizer", la.bpe.merges.length > 0 && la.opt != null && la.opt.step == 6, "step " + la.opt.step);

            System.out.println("== byte-resident inference ==");
            {
                Config[] rc = {
                    new Config(16, 4, 2, 8, 256, NORM_LN, ACT_GELU, POS_LEARNED),
                    new Config(16, 4, 2, 8, 256, NORM_RMS, ACT_SWIGLU, POS_ROPE),
                    new Config(24, 2, 1, 10, 256, NORM_LN, ACT_SWIGLU, POS_LEARNED)};
                Bpe plainB = new Bpe(new int[0][]);
                double worst = 0, worstFast = 0, worstInc = 0;
                boolean sameGen = true, sameBytes = true;
                int cases = 0;
                for (Config cfg : rc)
                    for (boolean tern : new boolean[]{false, true})
                        for (int grp : new int[]{7, 64}) {
                            Model fp = new Model(cfg, 21 + cases);
                            for (int i = 0; i < fp.w.length; i++) fp.w[i] *= 8; // sharpen outputs so the comparison is sensitive
                            Path fq = dir.resolve("r" + cases + ".q");
                            if (tern) saveTernary(fq.toString(), fp, plainB, grp); else saveQuant(fq.toString(), fp, plainB, grp);
                            Model deq = loadCkpt(fq.toString(), false).model;
                            Model res = loadCkpt(fq.toString(), true).model;
                            Model mem = residentFrom(fp, grp, tern);
                            sameBytes &= Arrays.equals(res.qw.q, mem.qw.q) && Arrays.equals(res.qw.scale, mem.qw.scale) && Arrays.equals(res.w, mem.w);
                            int T = cfg.ctx;
                            int[] tk = new int[T];
                            Rng rr2 = new Rng(100 + cases);
                            for (int i = 0; i < T; i++) tk[i] = (int) (rr2.nextDouble() * 256);
                            Cache ca = new Cache(cfg), cb = new Cache(cfg);
                            forward(deq, ca, tk, null, 0, T, null, 0);
                            forward(res, cb, tk, null, 0, T, null, 0);
                            int V = cfg.vocab;
                            for (int i = 0; i < V; i++) { double dd = Math.abs(ca.probs[(T - 1) * V + i] - cb.probs[(T - 1) * V + i]); if (grp == 7) worst = Math.max(worst, dd); else worstFast = Math.max(worstFast, dd); }
                            Cache cc = new Cache(cfg);
                            for (int u = 1; u <= T; u++) forward(res, cc, tk, null, u - 1, u, null, 0);
                            for (int i = 0; i < V; i++) worstInc = Math.max(worstInc, Math.abs(cc.probs[(T - 1) * V + i] - cb.probs[(T - 1) * V + i]));
                            sameGen &= Arrays.equals(
                                generateBytes(deq, plainB, "ab".getBytes(), 30, 0, 0, new Rng(1)),
                                generateBytes(res, plainB, "ab".getBytes(), 30, 0, 0, new Rng(1)));
                            cases++;
                        }
                report("resident forward == dequantized forward, exact kernel (group 7; int8/ternary x 3 configs)",
                        worst < 1e-9, String.format("max prob diff %.2e", worst));
                report("resident forward ~= dequantized forward, float row-scale kernel (group 64/0; int8/ternary x 3 configs)",
                        worstFast < 1e-5, String.format("max prob diff %.2e (float accumulation)", worstFast));
                { Model fp2 = new Model(rc[1], 5); for (int i = 0; i < fp2.w.length; i++) fp2.w[i] *= 8; Model r1 = residentFrom(fp2, 64, false), r2 = residentFrom(fp2, 64, false); r2.qw.q[r2.qw.q.length / 2] += 5; int[] tk2 = {1, 2, 3, 4, 5, 6, 7, 8}; Cache x1 = new Cache(rc[1]), x2 = new Cache(rc[1]); forward(r1, x1, tk2, null, 0, 8, null, 0); forward(r2, x2, tk2, null, 0, 8, null, 0); double sens = 0; for (int i = 0; i < 256; i++) sens = Math.max(sens, Math.abs(x1.probs[7 * 256 + i] - x2.probs[7 * 256 + i])); report("comparison is sensitive (one corrupted weight is detected)", sens > 1e-6, String.format("diff %.2e", sens)); }
                report("resident incremental == resident full forward", worstInc < 1e-12, String.format("max prob diff %.2e", worstInc));
                report("resident greedy generation == dequantized", sameGen, "");
                report("file-resident == in-memory residentFrom (bytes, scales, aux)", sameBytes, "");
                Model fpa = la.model;
                Model ra = residentFrom(fpa, 64, false);
                report("resident weights use <1/4 the memory of doubles", weightBytes(ra) * 4 < weightBytes(fpa), weightBytes(ra) + " vs " + weightBytes(fpa) + " bytes");
                Model rt = residentFrom(fpa, 64, true);
                report("resident ternary values are only -1, 0, +1", java.util.stream.IntStream.range(0, rt.qw.q.length).allMatch(i -> rt.qw.q[i] >= -1 && rt.qw.q[i] <= 1), "");
                saveQuant(dir.resolve("ra.q8").toString(), fpa, la.bpe, 64);
                Loaded lr = loadCkpt(dir.resolve("ra.q8").toString(), true);
                report("resident model carries no matrix doubles", lr.model.w.length < fpa.w.length / 4, lr.model.w.length + " vs " + fpa.w.length + " doubles");
            }

            System.out.println("== threaded inference ==");
            {
                Config tc = new Config(128, 4, 2, 32, 256, NORM_RMS, ACT_SWIGLU, POS_ROPE);
                Model base = new Model(tc, 77);
                Model[] variants = {base, residentFrom(base, 64, false), residentFrom(base, 64, true), residentFrom(base, 7, false), residentFrom(base, 0, false), residentFrom(base, 0, true)};
                String[] vn = {"double", "int8", "ternary", "int8 group 7", "int8 row-scale", "ternary row-scale"};
                int[] tk = new int[16];
                for (int i = 0; i < tk.length; i++) tk[i] = (i * 37 + 11) % 256;
                long before = parCalls.get();
                for (int v = 0; v < variants.length; v++) {
                    inferThreads = 1;
                    Cache c1 = new Cache(tc);
                    forward(variants[v], c1, tk, null, 0, tk.length, null, 0);
                    inferThreads = 5;
                    Cache c2 = new Cache(tc);
                    forward(variants[v], c2, tk, null, 0, tk.length, null, 0);
                    inferThreads = 1;
                    boolean same = true;
                    for (int i = 0; i < 16 * 256; i++) if (c1.probs[i] != c2.probs[i]) same = false;
                    report("threaded forward is bit-identical to single-threaded (" + vn[v] + ")", same, "");
                }
                report("the parallel path actually ran", parCalls.get() > before, (parCalls.get() - before) + " parallel matmuls");
            }

            System.out.println("== special tokens and held-out evaluation ==");
            {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < 150; i++) sb.append("abc<|endoftext|>\n");
                Path ep = dir.resolve("eos.txt");
                Files.write(ep, sb.toString().getBytes(StandardCharsets.UTF_8));
                HashMap<String, Long> ef = Bpe.wordFreq(ep, 2);
                boolean leak = false;
                for (String k : ef.keySet()) if (k.contains("endoftext") || k.contains("<|")) leak = true;
                report("BPE training never sees special-token text (no merges across or inside them)", !leak && ef.containsKey("abc"), "");
                Bpe sp = Bpe.train(ef, 260, 2), plainSp = new Bpe(new int[0][]);
                int eos = sp.eosId();
                byte[] doc = "ab<|endoftext|>abc<|pad|>x<|unk|>".getBytes(StandardCharsets.UTF_8);
                int[] enc = sp.encode(doc);
                int nEos = 0;
                for (int id : enc) if (id == eos) nEos++;
                report("special tokens: ids sit after merges, vocab = 256 + merges + specials",
                        eos == 256 + sp.merges.length && sp.vocabSize() == 256 + sp.merges.length + 2 && sp.specialId(1) == eos + 1 && sp.specialId(2) == -1, "eos id " + eos);
                report("special tokens: literal text becomes exactly one id each", nEos == 1 && Arrays.stream(enc).filter(i -> i >= eos).count() == 2 && sp.encode("<|unk|>".getBytes()).length == 7, "(only the 2 enabled specials are special)");
                report("special tokens: encode/decode round trip", Arrays.equals(sp.decode(enc, enc.length), doc), "");
                int[] a1 = sp.encode("abab".getBytes()), a2 = sp.encode("ab<|endoftext|>ab".getBytes());
                boolean noMerge = a2.length == 2 * (a1.length / 2) + 1 && a2[a1.length / 2] == eos;
                report("special tokens: pieces are encoded independently on either side", noMerge, "");
                report("without specials the same text is ordinary bytes", plainSp.encode("<|endoftext|>".getBytes()).length == 13, "");

                HashMap<String, String> se = baseArgs(dir, "eos.bin");
                se.put("data", ep.toString()); se.put("vocab", "256"); se.put("specials", "1");
                se.put("steps", "300"); se.put("eval", "300"); se.put("dropout", "0"); se.put("dim", "32");
                trainCore(TrainParams.from(se));
                Loaded le = loadCkpt(dir.resolve("eos.bin").toString());
                report("train specials=1: checkpoint round-trips the special-token count",
                        le.bpe.nSpecial == 1 && le.bpe.vocabSize() == 257 && le.model.cfg.vocab == 257, "vocab " + le.model.cfg.vocab);
                saveQuant(dir.resolve("eos.q8").toString(), le.model, le.bpe, 0);
                saveTernary(dir.resolve("eos.t").toString(), le.model, le.bpe, 0);
                report("quantized files keep the special-token count",
                        loadCkpt(dir.resolve("eos.q8").toString(), true).bpe.nSpecial == 1 && loadCkpt(dir.resolve("eos.t").toString()).bpe.eosId() == 256, "");
                byte[] g0 = generateBytes(le.model, le.bpe, "ab".getBytes(), 40, 0, 0, new Rng(1), false, false);
                byte[] g1 = generateBytes(le.model, le.bpe, "ab".getBytes(), 40, 0, 0, new Rng(1), false, true);
                report("generation stops at the end-of-text token when asked", g1.length < g0.length && new String(g1).startsWith("abc") && !new String(g1).contains("<|"),
                        "stopped output \"" + new String(g1).replace("\n", "\\n") + "\", free-running length " + g0.length);

                Config hc = new Config(16, 2, 2, 8, 256, NORM_RMS, ACT_SWIGLU, POS_ROPE);
                Model hm = new Model(hc, 9);
                int N = 61, V = 256;
                int[] hx = new int[N];
                Rng hr = new Rng(4);
                for (int i = 0; i < N; i++) hx[i] = (int) (hr.nextDouble() * 256);
                double[] sl1 = slidingLoss(hm, new Cache(hc), hx, 1);
                double ref = 0;
                for (int g = 1; g < N; g++) {
                    int s = Math.max(0, g - hc.ctx);
                    int[] w = Arrays.copyOfRange(hx, s, s + hc.ctx);
                    Cache cc = new Cache(hc);
                    forward(hm, cc, w, new int[hc.ctx], 0, hc.ctx, null, 0);
                    ref -= Math.log(cc.probs[(g - 1 - s) * V + hx[g]]);
                }
                report("held-out: stride 1 equals brute-force per-token loss (full context each token)", Math.abs(sl1[0] - ref / (N - 1)) < 1e-9 && (long) sl1[1] == N - 1,
                        String.format("%.6f vs %.6f", sl1[0], ref / (N - 1)));
                boolean once = true;
                for (int st : new int[]{1, 2, 3, 5, 8, 100}) once &= (long) slidingLoss(hm, new Cache(hc), hx, st)[1] == N - 1;
                double[] sl8 = slidingLoss(hm, new Cache(hc), hx, 8), sl4 = slidingLoss(hm, new Cache(hc), hx, 4);
                report("held-out: every token scored exactly once for any stride", once, "");
                report("held-out: larger stride (less context) never changes the token count and stays finite", Double.isFinite(sl8[0]) && Double.isFinite(sl4[0]) && Math.abs(sl8[0] - sl1[0]) < 0.5, String.format("stride 1/4/8: %.4f %.4f %.4f", sl1[0], sl4[0], sl8[0]));
            }
            System.out.println("== int8 quantization ==");
            {
                Path qp = dir.resolve("a.q8");
                saveQuant(qp.toString(), la.model, la.bpe, 16);
                Loaded lq = loadCkpt(qp.toString());
                double worst = 0;
                boolean boundOk = true;
                for (int[] r : la.model.lay.decay)
                    for (int g = r[0]; g < r[1]; g += 16) {
                        int end = Math.min(r[1], g + 16);
                        double mx = 0;
                        for (int i = g; i < end; i++) mx = Math.max(mx, Math.abs(la.model.w[i]));
                        double bound = mx / 127.0 * 0.5 * (1 + 1e-6) + 1e-12;
                        for (int i = g; i < end; i++) {
                            double err = Math.abs(la.model.w[i] - lq.model.w[i]);
                            worst = Math.max(worst, err);
                            if (err > bound) boundOk = false;
                        }
                    }
                report("int8 error <= scale/2 for every weight", boundOk, String.format("max abs err %.2e", worst));
                boolean[] mat = matrixMask(la.model.lay);
                boolean exact = true;
                for (int i = 0; i < mat.length; i++) if (!mat[i] && la.model.w[i] != lq.model.w[i]) exact = false;
                report("norms and biases stay exact", exact, "");
                report("int8 file is smaller", Files.size(qp) * 3 < Files.size(dir.resolve("a.bin")) , Files.size(qp) + " vs " + Files.size(dir.resolve("a.bin")) + " bytes (includes fp64 Adam state)");
                Cache c0 = new Cache(la.model.cfg), c1 = new Cache(lq.model.cfg);
                int T = la.model.cfg.ctx;
                int[] tk = new int[T];
                Rng tr = new Rng(2);
                for (int i = 0; i < T; i++) tk[i] = (int) (tr.nextDouble() * 256);
                forward(la.model, c0, tk, null, 0, T, null, 0);
                forward(lq.model, c1, tk, null, 0, T, null, 0);
                int V = la.model.cfg.vocab;
                double tv = 0;
                for (int i = 0; i < V; i++) tv += Math.abs(c0.probs[(T - 1) * V + i] - c1.probs[(T - 1) * V + i]);
                report("quantized forward close to fp64", tv / 2 < 0.05, String.format("total variation %.2e", tv / 2));
                report("quantized model still generates", generateBytes(lq.model, lq.bpe, "ab".getBytes(), 20, 0.8, 20, new Rng(1)).length >= 22, "");
            }

            HashMap<String, String> b1 = baseArgs(dir, "b.bin");
            b1.put("steps", "12"); b1.put("until", "6");
            trainCore(TrainParams.from(b1));
            HashMap<String, String> b2 = baseArgs(dir, "b.bin");
            b2.put("steps", "12"); b2.put("resume", dir.resolve("b.bin").toString());
            Model mb = trainCore(TrainParams.from(b2));
            HashMap<String, String> full = baseArgs(dir, "full.bin");
            full.put("steps", "12");
            Model mf = trainCore(TrainParams.from(full));
            Loaded lb = loadCkpt(dir.resolve("b.bin").toString()), lf = loadCkpt(dir.resolve("full.bin").toString());
            report("resume is bit-exact (weights)", maxDiff(mb.w, mf.w) == 0, String.format("diff %.1e", maxDiff(mb.w, mf.w)));
            report("resume is bit-exact (Adam state)", maxDiff(lb.opt.m, lf.opt.m) == 0 && maxDiff(lb.opt.v, lf.opt.v) == 0 && lb.opt.step == lf.opt.step, "");

            HashMap<String, String> ac = baseArgs(dir, "acc.bin");
            ac.put("batch", "2"); ac.put("accum", "2");
            Model mc = trainCore(TrainParams.from(ac));
            report("accumulation (2x2) == batch 4", maxDiff(ma.w, mc.w) < 1e-7, String.format("max weight diff %.2e", maxDiff(ma.w, mc.w)));

            HashMap<String, String> st = baseArgs(dir, "stream.bin");
            st.put("stream", "1");
            Model ms = trainCore(TrainParams.from(st));
            report("streaming == in-memory (BPE tokens)", maxDiff(ma.w, ms.w) == 0, "");

            Files.write(dir.resolve("bytes.txt"), text);
            HashMap<String, String> by = baseArgs(dir, "by.bin");
            by.put("data", dir.resolve("bytes.txt").toString()); by.put("vocab", "256");
            Model m1 = trainCore(TrainParams.from(by));
            by.put("stream", "1"); by.put("out", dir.resolve("by2.bin").toString());
            Model m2 = trainCore(TrainParams.from(by));
            report("streaming == in-memory (bytes)", maxDiff(m1.w, m2.w) == 0, "");

            HashMap<String, String> lr = baseArgs(dir, "learn.bin");
            lr.put("steps", "300"); lr.put("eval", "300"); lr.put("dropout", "0"); lr.put("dim", "32");
            lr.put("batch", "8"); lr.put("ctx", "24");
            trainCore(TrainParams.from(lr));
            Loaded ll = loadCkpt(dir.resolve("learn.bin").toString());
            try (Corpus cp = Corpus.open(dir.resolve("learn.bin.tokens"), true, false)) {
                double vl = evalLoss(ll.model, new Cache(ll.model.cfg), cp, 0, cp.length(), 5, 16);
                double uniform = Math.log(ll.model.cfg.vocab);
                report("model learns (loss << ln(vocab))", vl < 0.5 * uniform, String.format("loss %.3f vs uniform %.3f", vl, uniform));
            }
        } finally {
            try (var s = Files.walk(dir)) {
                s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
        tally();
    }

    /** Straightforward O(merges * corpus) trainer used only to cross-check the incremental one. */
    static int[][] naiveBpe(HashMap<String, Long> freq, int vocab) {
        List<String> keys = new ArrayList<>(freq.keySet());
        Collections.sort(keys);
        List<int[]> words = new ArrayList<>();
        List<Long> cnt = new ArrayList<>();
        for (String k : keys) {
            int[] ids = new int[k.length()];
            for (int i = 0; i < ids.length; i++) ids[i] = k.charAt(i);
            words.add(ids);
            cnt.add(freq.get(k));
        }
        List<int[]> out = new ArrayList<>();
        for (int m = 0; m < vocab - 256; m++) {
            TreeMap<Long, Long> stats = new TreeMap<>();
            for (int w = 0; w < words.size(); w++) {
                int[] ids = words.get(w);
                for (int j = 0; j + 1 < ids.length; j++) stats.merge(Bpe.key(ids[j], ids[j + 1]), cnt.get(w), Long::sum);
            }
            long bk = -1, bc = 0;
            for (Map.Entry<Long, Long> e : stats.entrySet()) if (e.getValue() > bc) { bc = e.getValue(); bk = e.getKey(); }
            if (bc <= 0) break;
            int a = (int) (bk >>> 32), b = (int) bk;
            for (int w = 0; w < words.size(); w++) words.set(w, Bpe.mergePair(words.get(w), a, b, 256 + m));
            out.add(new int[]{a, b});
        }
        return out.toArray(new int[0][]);
    }

    static void tally() {
        System.out.printf("%n%d passed, %d failed%n", passes, fails);
        if (fails > 0) System.exit(1);
    }

    static String need(Map<String, String> a, String k) {
        String v = a.get(k);
        if (v == null) throw new IllegalArgumentException("Missing argument " + k + "=...");
        return v;
    }
    static int geti(Map<String, String> a, String k, int d) { return a.containsKey(k) ? Integer.parseInt(a.get(k)) : d; }
    static double getd(Map<String, String> a, String k, double d) { return a.containsKey(k) ? Double.parseDouble(a.get(k)) : d; }
    static boolean getf(Map<String, String> a, String k) { String v = a.get(k); return v != null && !v.equals("0") && !v.equalsIgnoreCase("false"); }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("SingleFileGPT — pure-Java zero-dependency decoder-only transformer");
            System.out.println("Commands: selftest | train data=... out=... | generate model=... prompt=... | quantize model=... out=... | eval model=... data=... | heldout model=... data=... | bench");
            return;
        }
        Map<String, String> a = new HashMap<>();
        for (int i = 1; i < args.length; i++) {
            int eq = args[i].indexOf('=');
            if (eq > 0) a.put(args[i].substring(0, eq), args[i].substring(eq + 1));
        }
        switch (args[0]) {
            case "selftest": selftest(); break;
            case "train": trainCore(TrainParams.from(a)); break;
            case "generate": generate(a); break;
            case "quantize": quantizeCmd(a); break;
            case "eval": evalCmd(a); break;
            case "heldout": heldoutCmd(a); break;
            case "bench": benchCmd(a); break;
            default: throw new IllegalArgumentException("Unknown command " + args[0]);
        }
    }
}
