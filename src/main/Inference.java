// having feature that sentiment of a phrase and baseform of each individual word present in the phrase is being taken into account.

// this code is for all binary features and word trees to phrase trees conversion
// not supressing anything
// 'not' function word negator problem that whether they will always be combined with some other word or not, how to take their effect.
// some problem with phrase trees when there is possibility of combination after two combinations but it will serve the basic conversion to phrase trees

package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.io.StringReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import com.clearnlp.component.AbstractComponent;
import com.clearnlp.dependency.DEPTree;
import com.clearnlp.nlp.NLPGetter;
import com.clearnlp.nlp.NLPMode;
import com.clearnlp.reader.AbstractReader;
import com.clearnlp.segmentation.AbstractSegmenter;
import com.clearnlp.tokenization.AbstractTokenizer;

import edu.umass.cs.mallet.grmm.inference.Inferencer;
import edu.umass.cs.mallet.grmm.inference.JunctionTreeInferencer;
import edu.umass.cs.mallet.grmm.types.Assignment;
import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.FactorGraph;
import edu.umass.cs.mallet.grmm.types.HashVarSet;
import edu.umass.cs.mallet.grmm.types.TableFactor;
import edu.umass.cs.mallet.grmm.types.VarSet;
import edu.umass.cs.mallet.grmm.types.Variable;

public class Inference {
	final String language = AbstractReader.LANG_EN, modelType = "general-en";

	AbstractTokenizer tokenizer;
	AbstractComponent tagger, parser, identifier, classifier, labeler;

	public Inference() throws Exception {
		tokenizer = NLPGetter.getTokenizer(language);
		tagger = NLPGetter.getComponent(modelType, language, NLPMode.MODE_POS);
		parser = NLPGetter.getComponent(modelType, language, NLPMode.MODE_DEP);
		identifier = NLPGetter.getComponent(modelType, language, NLPMode.MODE_PRED);
		classifier = NLPGetter.getComponent(modelType, language, NLPMode.MODE_ROLE);
		labeler = NLPGetter.getComponent(modelType, language, NLPMode.MODE_SRL);
	}

	public String[][][] get(String sent) throws Exception {
		AbstractComponent[] components = { tagger, parser, identifier, classifier, labeler };
		if (sent == null)
			return new String[0][0][0];
		BufferedReader reader = new BufferedReader(new StringReader(sent));
		return process(tokenizer, components, reader);
	}

	String[][][] process(AbstractTokenizer tokenizer, AbstractComponent[] components, BufferedReader reader)
			throws Exception {
		AbstractSegmenter segmenter = NLPGetter.getSegmenter(language, tokenizer);
		DEPTree tree;
		List<List<String>> sentences = segmenter.getSentences(reader);
		String[][][] s = new String[sentences.size()][][];
		int idx = 0;

		for (List<String> tokens : sentences) {
			tree = NLPGetter.toDEPTree(tokens);
			for (AbstractComponent component : components)
				component.process(tree);
			String[] r = tree.toStringDEP().split("\n");
			s[idx] = new String[r.length][];
			for (int i = 0; i < r.length; i++) {
				s[idx][i] = r[i].split("\t");
			}
			idx++;
		}

		return s;
	}

	public static int feature0(int si) {
		return si;
	}

	public static int isequal(int one, int two) {
		if (one == two)
			return 1;
		else
			return 0;
	}

	public static int isequal(String s1, String s2) {
		if (s1.equals(s2))
			return 1;
		else
			return 0;
	}

	public static int issamepol(int pri, int sentiment) {
		if (pri == 0)
			return 0;
		if (pri != sentiment)
			return -1;
		else
			return 0;
	}

	public static void multiply(double[][] mata, double[][] matb, double[][] matc) {
		for (int a = 0; a < mata.length; a++) {
			for (int b = 0; b < matb[0].length; b++) {
				for (int c = 0; c < matb.length; c++) {
					matc[a][b] += mata[a][c] * matb[c][b];
				}
			}
		}
	}

	public static void identity(double[][] mat) {
		for (int a = 0; a < mat.length; a++) {
			mat[a][a] = 1.0;
		}
	}

	public static void scalar(double[][] mat, double alpha) {
		for (int a = 0; a < mat.length; a++) {
			for (int b = 0; b < mat[0].length; b++) {
				mat[a][b] *= alpha;
			}
		}
	}

	public static void addition(double[][] mata, double[][] matb, double[][] matc) {
		for (int a = 0; a < mata.length; a++) {
			for (int b = 0; b < mata[0].length; b++) {
				matc[a][b] = mata[a][b] + matb[a][b];
			}
		}
	}

	public static void equate(double[][] mata, double[][] matb) {
		for (int a = 0; a < mata.length; a++) {
			for (int b = 0; b < mata[0].length; b++) {
				mata[a][b] = matb[a][b];
			}
		}
	}

	public static void transpose(double[][] mata, double[][] matb) {
		for (int a = 0; a < matb.length; a++) {
			for (int b = 0; b < matb[0].length; b++) {
				matb[a][b] = mata[b][a];
			}
		}
	}

	public static void norm(double[][] gradnew) {
		double acc = 0;
		for (int i = 0; i < gradnew.length; i++) {
			acc += Math.abs(gradnew[i][0]);
		}
		scalar(gradnew, (1.0 / acc));

	}

	public static void converter(double[] si, double[][] siqi, double[][][] siqiri, double[][] sifi, double[][] sibi,
			double[][] sisj, double[][][] sisjrj, double[][][][] sisjrjqj, double[][][] sisjbi, double[][][] sisjbj,
			double[][] params) {
		int count = 0;
		for (int i = 0; i < si.length; i++)
			si[i] = params[count++][0];

		for (int i = 0; i < siqi.length; i++)
			for (int j = 0; j < siqi[0].length; j++)
				siqi[i][j] = params[count++][0];

		for (int i = 0; i < siqiri.length; i++)
			for (int j = 0; j < siqiri[0].length; j++)
				for (int k = 0; k < siqiri[0][0].length; k++)
					siqiri[i][j][k] = params[count++][0];

		for (int i = 0; i < sifi.length; i++)
			for (int j = 0; j < sifi[0].length; j++)
				sifi[i][j] = params[count++][0];

		for (int i = 0; i < sibi.length; i++)
			for (int j = 0; j < sibi[0].length; j++)
				sibi[i][j] = params[count++][0];

		for (int i = 0; i < sisj.length; i++)
			for (int j = 0; j < sisj[0].length; j++)
				sisj[i][j] = params[count++][0];

		for (int i = 0; i < sisjrj.length; i++)
			for (int j = 0; j < sisjrj[0].length; j++)
				for (int k = 0; k < sisjrj[0][0].length; k++)
					sisjrj[i][j][k] = params[count++][0];

		for (int i = 0; i < sisjrjqj.length; i++)
			for (int j = 0; j < sisjrjqj[0].length; j++)
				for (int k = 0; k < sisjrjqj[0][0].length; k++)
					for (int l = 0; l < sisjrjqj[0][0][0].length; l++)
						sisjrjqj[i][j][k][l] = params[count++][0];

		for (int i = 0; i < sisjbi.length; i++)
			for (int j = 0; j < sisjbi[0].length; j++)
				for (int k = 0; k < sisjbi[0][0].length; k++)
					sisjbi[i][j][k] = params[count++][0];

		for (int i = 0; i < sisjbj.length; i++)
			for (int j = 0; j < sisjbj[0].length; j++)
				for (int k = 0; k < sisjbj[0][0].length; k++)
					sisjbj[i][j][k] = params[count++][0];

	}

	public static double marginalise(String[][] deptree, int[] phrase, int nphrases, double[] si, double[][] siqi,
			double[][][] siqiri, double[][] sifi, double[][] sibi, double[][] sisj, double[][][] sisjrj,
			double[][][][] sisjrjqj, double[][][] sisjbi, double[][][] sisjbj, double[][] params, int polarity,
			boolean first) {

		int[][] ptree = new int[nphrases][2];
		int[] phraseno = new int[deptree.length];
		int mycount = 0;
		for (int i = 0; i < deptree.length; i++) {
			ptree[mycount][0] = i;
			for (int j = i; j < deptree.length; j++) {
				if (phrase[j] != phrase[i]) {
					ptree[mycount][1] = j - 1;
					i = j - 1;
					break;
				} else {
					phraseno[j] = mycount;
					if (j == deptree.length - 1)
						i = j;
				}
			}

			mycount += 1;
		}

		Variable[] allVars = new Variable[nphrases + 1];
		for (int i = 0; i < nphrases + 1; i++) {
			allVars[i] = new Variable(2);
		}
		FactorGraph mdl = new FactorGraph(allVars);

		int[] priorpol = new int[nphrases];
		int[] reversepol = new int[nphrases];

		for (int i = 0; i < nphrases; i++) {
			String bi = deptree[ptree[i][1]][2];
			int priorpolarity = 0;
			if (priorSent.get(bi) != null)
				priorpolarity = priorSent.get(bi);
			priorpol[i] = priorpolarity;

			boolean reverse = false;
			// boolean reverse = polrev.contains(bi);
			for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
				if (polrev.contains(deptree[j][2])) {
					reverse = true;
					break;
				}
			}

			if (reverse == false)
				reversepol[i] = 0;
			else
				reversepol[i] = 1;

		}

		for (int i = 0; i < nphrases; i++) {
			for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
				String bi = deptree[j][2];
				int head = Integer.parseInt(deptree[j][5]);
				if (head == 0 && fNeg.contains(bi)) {
					priorpol[i] = -1 * priorpol[i];
					reversepol[i] = 1 - reversepol[i];
				} else if (fNeg.contains(bi)) {
					int start = phrase[head - 1];
					int k = 0;
					while (k < nphrases) {
						if (ptree[k][0] == start)
							break;
						k = k + 1;
					}
					priorpol[k] = -1 * priorpol[k];
					reversepol[k] = 1 - reversepol[k];
				}
			}

		}
		// Add Node Factors , starting the loop from the index 1 as there is no
		// node factor for the root virtual node, combine these features for a
		// word
		for (int i = 0; i < nphrases; i++) {

			double ptl10 = Math.exp(si[0]);
			double ptl11 = Math.exp(si[1]);

			double ptl20 = Math.exp(siqi[0][priorpol[i] + 1]);
			double ptl21 = Math.exp(siqi[1][priorpol[i] + 1]);

			double ptl30 = Math.exp(siqiri[0][priorpol[i] + 1][reversepol[i]]);
			double ptl31 = Math.exp(siqiri[1][priorpol[i] + 1][reversepol[i]]);

			double ptl40 = 1;
			double ptl41 = 1;

			double ptl50 = 1;
			double ptl51 = 1;

			for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {

				if (j < ptree[i][1]) {
					String bij = deptree[j][2];
					String bij1 = deptree[j + 1][2];
					if ((baseform.get(bij) != null)) {
						ptl50 = ptl50 * Math.exp(sibi[0][baseform.get(bij) - 1]);
						ptl51 = ptl51 * Math.exp(sibi[1][baseform.get(bij) - 1]);
					}
				}

				String fij = deptree[j][3];

				if (postag.get(fij) != null) {
					ptl40 = ptl40 * Math.exp(sifi[0][postag.get(fij) - 1]);
					ptl41 = ptl41 * Math.exp(sifi[1][postag.get(fij) - 1]);
				}

			}

			double nodevalue0 = ptl10 * ptl20 * ptl30 * ptl40 * ptl50;
			double nodevalue1 = ptl11 * ptl21 * ptl31 * ptl41 * ptl51;
			double[] ptlTable = { nodevalue0, nodevalue1 };

			mdl.addFactor(new TableFactor((VarSet) new HashVarSet(new Variable[] { allVars[i + 1] }), ptlTable));

			// System.out.println("adding node feature");
		}

		// Add Edge Factors
		for (int i = 0; i < nphrases; i++) {

			int head;
			int headphrase = 0;
			boolean found = false;
			for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
				head = Integer.parseInt(deptree[j][5]);
				if (head != 0)
					if (phraseno[head - 1] != i) {
						headphrase = phraseno[head - 1];
						found = true;
						break;
					}
			}

			if (!found) {
				double ptl100 = Math.exp(sisj[0][0]);
				double ptl101 = Math.exp(sisj[0][1]);
				double ptl110 = Math.exp(sisj[1][0]);
				double ptl111 = Math.exp(sisj[1][1]);
				double[] ptlTable = { ptl100, ptl101, ptl110, ptl111 };

				for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
					String bij = deptree[j][2];
					if (baseform.get(bij) != null) {
						int index = baseform.get(bij);
						ptlTable[0] = ptlTable[0] * Math.exp(sisjbi[0][0][index - 1]);
						ptlTable[1] = ptlTable[1] * Math.exp(sisjbi[0][1][index - 1]);
						ptlTable[2] = ptlTable[2] * Math.exp(sisjbi[1][0][index - 1]);
						ptlTable[3] = ptlTable[3] * Math.exp(sisjbi[1][1][index - 1]);

					}

				}

				mdl.addFactor(new TableFactor((VarSet) new HashVarSet(new Variable[] { allVars[i + 1], allVars[0] }),
						ptlTable));
			}

			else {

				double ptl100 = Math.exp(sisj[0][0]);
				double ptl101 = Math.exp(sisj[0][1]);
				double ptl110 = Math.exp(sisj[1][0]);
				double ptl111 = Math.exp(sisj[1][1]);

				double ptl200 = Math.exp(sisjrj[0][0][reversepol[headphrase]]);
				double ptl201 = Math.exp(sisjrj[0][1][reversepol[headphrase]]);
				double ptl210 = Math.exp(sisjrj[1][0][reversepol[headphrase]]);
				double ptl211 = Math.exp(sisjrj[1][1][reversepol[headphrase]]);

				double ptl300 = Math.exp(sisjrjqj[0][0][reversepol[headphrase]][priorpol[headphrase] + 1]);
				double ptl301 = Math.exp(sisjrjqj[0][1][reversepol[headphrase]][priorpol[headphrase] + 1]);
				double ptl310 = Math.exp(sisjrjqj[1][0][reversepol[headphrase]][priorpol[headphrase] + 1]);
				double ptl311 = Math.exp(sisjrjqj[1][1][reversepol[headphrase]][priorpol[headphrase] + 1]);

				double evalue00 = ptl100 * ptl200 * ptl300;
				double evalue01 = ptl101 * ptl201 * ptl301;
				double evalue10 = ptl110 * ptl210 * ptl310;
				double evalue11 = ptl111 * ptl211 * ptl311;

				double[] ptlTable = { evalue00, evalue01, evalue10, evalue11 };

				for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
					String bij = deptree[j][2];
					if (baseform.get(bij) != null) {
						int index = baseform.get(bij);
						ptlTable[0] = ptlTable[0] * Math.exp(sisjbi[0][0][index - 1]);
						ptlTable[1] = ptlTable[1] * Math.exp(sisjbi[0][1][index - 1]);
						ptlTable[2] = ptlTable[2] * Math.exp(sisjbi[1][0][index - 1]);
						ptlTable[3] = ptlTable[3] * Math.exp(sisjbi[1][1][index - 1]);
					}

				}

				for (int k = ptree[headphrase][0]; k <= ptree[headphrase][1]; k++) {
					String bjk = deptree[k][2];
					if (baseform.get(bjk) != null) {
						int index = baseform.get(bjk);
						ptlTable[0] = ptlTable[0] * Math.exp(sisjbj[0][0][index - 1]);
						ptlTable[1] = ptlTable[1] * Math.exp(sisjbj[0][1][index - 1]);
						ptlTable[2] = ptlTable[2] * Math.exp(sisjbj[1][0][index - 1]);
						ptlTable[3] = ptlTable[3] * Math.exp(sisjbj[1][1][index - 1]);

					}

				}

				mdl.addFactor(new TableFactor(
						(VarSet) new HashVarSet(new Variable[] { allVars[i + 1], allVars[headphrase + 1] }), ptlTable));
				// System.out.println("adding edge feature");
			}
		}
		Inferencer inf = new JunctionTreeInferencer(); // combine different edge
		// features of a node,
		// similarly node
		// features
		inf.computeMarginals(mdl);

		Factor ptl;
		Assignment assn;

		ptl = inf.lookupMarginal(allVars[0]); // ptl will be the corresponding
		// factor of the variable var
		// assignment iterator method is to query a factor ptl of the variable
		// so
		double normalise = ptl.value(ptl.assignmentIterator());

		if (polarity == -1) {

		} else {
			normalise = 1 - normalise;
		}

		double priors = 0;

		if (first) {
			for (int k = 0; k < params.length; k++) {
				priors += params[k][0] * params[k][0];
			}
		}

		return (Math.log(normalise) - (1.0 / 8) * priors);

	}

	// phrase is a vector to show phrase no. of each word in the sentence while
	// ptree is a two d array showing starting and ending location of a phrase
	// phraseno tells the phrase no. ith word in the sentence
	public static void createFactorGraph(String[][] deptree, int[] phrase, int nphrases, double[] si, double[][] siqi,
			double[][][] siqiri, double[][] sifi, double[][] sibi, double[][] sisj, double[][][] sisjrj,
			double[][][][] sisjrjqj, double[][][] sisjbi, double[][][] sisjbj, double[] sigrad, double[][] siqigrad,
			double[][][] siqirigrad, double[][] sifigrad, double[][] sibigrad, double[][] sisjgrad,
			double[][][] sisjrjgrad, double[][][][] sisjrjqjgrad, double[][][] sisjbigrad, double[][][] sisjbjgrad,
			int polarity) {

		int[][] ptree = new int[nphrases][2]; // stores start and end word index
		// of each phrase
		int[] phraseno = new int[deptree.length];
		// System.out.println("start");
		// System.out.println(nphrases);
		/*
		 * System.out.println("phrase array"); for(int
		 * my=0;my<phrase.length;my++){ System.out.println(phrase[my]); }
		 */

		int mycount = 0;
		for (int i = 0; i < deptree.length; i++) {
			// System.out.println(mycount);
			ptree[mycount][0] = i;
			for (int j = i; j < deptree.length; j++) {
				if (phrase[j] != phrase[i]) {
					ptree[mycount][1] = j - 1;
					i = j - 1;
					break;
				} else {
					phraseno[j] = mycount;
					if (j == deptree.length - 1)
						i = j;
				}
			}

			mycount += 1;
		}

		Variable[] allVars = new Variable[nphrases + 1];
		for (int i = 0; i < nphrases + 1; i++) {
			allVars[i] = new Variable(2);
		}
		FactorGraph mdl = new FactorGraph(allVars);

		int[] priorpol = new int[nphrases];
		int[] reversepol = new int[nphrases];

		for (int i = 0; i < nphrases; i++) {
			String bi = deptree[ptree[i][1]][2];
			int priorpolarity = 0;
			if (priorSent.get(bi) != null)
				priorpolarity = priorSent.get(bi);
			priorpol[i] = priorpolarity;

			boolean reverse = false;
			// boolean reverse = polrev.contains(bi);
			for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
				if (polrev.contains(deptree[j][2])) {
					reverse = true;
					break;
				}
			}

			if (reverse == false)
				reversepol[i] = 0;
			else
				reversepol[i] = 1;

		}

		for (int i = 0; i < nphrases; i++) {
			for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
				String bi = deptree[j][2];
				int head = Integer.parseInt(deptree[j][5]);
				if (head == 0 && fNeg.contains(bi)) {
					priorpol[i] = -1 * priorpol[i];
					reversepol[i] = 1 - reversepol[i];
				} else if (fNeg.contains(bi)) {

					int start = phrase[head - 1];
					// System.out.println("error start");
					// System.out.println(start);
					int k = 0;
					while (k < nphrases) {
						// System.out.println("error search");
						// System.out.println(ptree[k][0]);
						if (ptree[k][0] == start)
							break;
						k = k + 1;
					}
					priorpol[k] = -1 * priorpol[k];
					reversepol[k] = 1 - reversepol[k];
				}
			}

		}
		// Add Node Factors , starting the loop from the index 1 as there is no
		// node factor for the root virtual node, combine these features for a
		// word
		for (int i = 0; i < nphrases; i++) {

			double ptl10 = Math.exp(si[0]);
			double ptl11 = Math.exp(si[1]);

			double ptl20 = Math.exp(siqi[0][priorpol[i] + 1]);
			double ptl21 = Math.exp(siqi[1][priorpol[i] + 1]);

			double ptl30 = Math.exp(siqiri[0][priorpol[i] + 1][reversepol[i]]);
			double ptl31 = Math.exp(siqiri[1][priorpol[i] + 1][reversepol[i]]);

			double ptl40 = 1;
			double ptl41 = 1;

			double ptl50 = 1;
			double ptl51 = 1;

			for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {

				if (j < ptree[i][1]) {
					String bij = deptree[j][2];
					String bij1 = deptree[j + 1][2];
					if ((baseform.get(bij) != null)) {
						ptl50 = ptl50 * Math.exp(sibi[0][baseform.get(bij) - 1]);
						ptl51 = ptl51 * Math.exp(sibi[1][baseform.get(bij) - 1]);
					}
				}

				String fij = deptree[j][3];

				if (postag.get(fij) != null) {
					ptl40 = ptl40 * Math.exp(sifi[0][postag.get(fij) - 1]);
					ptl41 = ptl41 * Math.exp(sifi[1][postag.get(fij) - 1]);
				}

			}

			double nodevalue0 = ptl10 * ptl20 * ptl30 * ptl40 * ptl50;
			double nodevalue1 = ptl11 * ptl21 * ptl31 * ptl41 * ptl51;
			double[] ptlTable = { nodevalue0, nodevalue1 };

			mdl.addFactor(new TableFactor((VarSet) new HashVarSet(new Variable[] { allVars[i + 1] }), ptlTable));

			// System.out.println("adding node feature");
		}

		// Add Edge Factors
		for (int i = 0; i < nphrases; i++) {

			int head;
			int headphrase = 0;
			boolean found = false;
			for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
				head = Integer.parseInt(deptree[j][5]);
				if (head != 0)
					if (phraseno[head - 1] != i) {
						headphrase = phraseno[head - 1];
						found = true;
						break;
					}
			}

			if (!found) {
				double ptl100 = Math.exp(sisj[0][0]);
				double ptl101 = Math.exp(sisj[0][1]);
				double ptl110 = Math.exp(sisj[1][0]);
				double ptl111 = Math.exp(sisj[1][1]);
				double[] ptlTable = { ptl100, ptl101, ptl110, ptl111 };

				for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
					String bij = deptree[j][2];
					if (baseform.get(bij) != null) {
						int index = baseform.get(bij);
						ptlTable[0] = ptlTable[0] * Math.exp(sisjbi[0][0][index - 1]);
						ptlTable[1] = ptlTable[1] * Math.exp(sisjbi[0][1][index - 1]);
						ptlTable[2] = ptlTable[2] * Math.exp(sisjbi[1][0][index - 1]);
						ptlTable[3] = ptlTable[3] * Math.exp(sisjbi[1][1][index - 1]);

					}

				}

				mdl.addFactor(new TableFactor((VarSet) new HashVarSet(new Variable[] { allVars[i + 1], allVars[0] }),
						ptlTable));
			}

			else {

				double ptl100 = Math.exp(sisj[0][0]);
				double ptl101 = Math.exp(sisj[0][1]);
				double ptl110 = Math.exp(sisj[1][0]);
				double ptl111 = Math.exp(sisj[1][1]);

				double ptl200 = Math.exp(sisjrj[0][0][reversepol[headphrase]]);
				double ptl201 = Math.exp(sisjrj[0][1][reversepol[headphrase]]);
				double ptl210 = Math.exp(sisjrj[1][0][reversepol[headphrase]]);
				double ptl211 = Math.exp(sisjrj[1][1][reversepol[headphrase]]);

				double ptl300 = Math.exp(sisjrjqj[0][0][reversepol[headphrase]][priorpol[headphrase] + 1]);
				double ptl301 = Math.exp(sisjrjqj[0][1][reversepol[headphrase]][priorpol[headphrase] + 1]);
				double ptl310 = Math.exp(sisjrjqj[1][0][reversepol[headphrase]][priorpol[headphrase] + 1]);
				double ptl311 = Math.exp(sisjrjqj[1][1][reversepol[headphrase]][priorpol[headphrase] + 1]);

				double evalue00 = ptl100 * ptl200 * ptl300;
				double evalue01 = ptl101 * ptl201 * ptl301;
				double evalue10 = ptl110 * ptl210 * ptl310;
				double evalue11 = ptl111 * ptl211 * ptl311;

				double[] ptlTable = { evalue00, evalue01, evalue10, evalue11 };

				for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
					String bij = deptree[j][2];
					if (baseform.get(bij) != null) {
						int index = baseform.get(bij);
						ptlTable[0] = ptlTable[0] * Math.exp(sisjbi[0][0][index - 1]);
						ptlTable[1] = ptlTable[1] * Math.exp(sisjbi[0][1][index - 1]);
						ptlTable[2] = ptlTable[2] * Math.exp(sisjbi[1][0][index - 1]);
						ptlTable[3] = ptlTable[3] * Math.exp(sisjbi[1][1][index - 1]);
					}

				}

				for (int k = ptree[headphrase][0]; k <= ptree[headphrase][1]; k++) {
					String bjk = deptree[k][2];
					if (baseform.get(bjk) != null) {
						int index = baseform.get(bjk);
						ptlTable[0] = ptlTable[0] * Math.exp(sisjbj[0][0][index - 1]);
						ptlTable[1] = ptlTable[1] * Math.exp(sisjbj[0][1][index - 1]);
						ptlTable[2] = ptlTable[2] * Math.exp(sisjbj[1][0][index - 1]);
						ptlTable[3] = ptlTable[3] * Math.exp(sisjbj[1][1][index - 1]);

					}

				}

				mdl.addFactor(new TableFactor(
						(VarSet) new HashVarSet(new Variable[] { allVars[i + 1], allVars[headphrase + 1] }), ptlTable));
				// System.out.println("adding edge feature");
			}
		}
		Inferencer inf = new JunctionTreeInferencer(); // combine different edge
		// features of a node,
		// similarly node
		// features
		inf.computeMarginals(mdl);

		Factor ptl;
		Assignment assn;
		/*
		 * int head = Integer.parseInt(deptree[1][5]) ; Factor ptl =
		 * inf.lookupMarginal((VarSet)new HashVarSet(new
		 * Variable[]{allVars[2],allVars[head]})); //ptl will be the
		 * corresponding factor of the variable var //assignment iterator method
		 * is to query a factor ptl of the variable so AssignmentIterator it =
		 * ptl.assignmentIterator (); while(it.hasNext()){ System.out.println(
		 * "Marginals  " + ptl.value(it)); it.advance() ; }
		 * 
		 * Assignment assn = new Assignment(mdl, new int[]{0,0,0,1,1} );
		 * System.out.println("Assignment " +
		 * Math.exp(inf.lookupLogJoint(assn)));
		 * 
		 */

		double[] sigradvector = new double[2];
		double[][] siqigradvector = new double[2][3];
		double[][][] siqirigradvector = new double[2][3][2]; // surface form
		// feature is
		// being
		// omitted, also
		// no coarse
		// grained
		// postag is
		// there
		double[][] sifigradvector = new double[2][46];
		double[][] sibigradvector = new double[sibi.length][sibi[0].length];

		double[][] sisjgradvector = new double[2][2];
		double[][][] sisjrjgradvector = new double[2][2][2];
		double[][][][] sisjrjqjgradvector = new double[2][2][2][3];
		double[][][] sisjbigradvector = new double[2][2][sisjbi[0][0].length];
		double[][][] sisjbjgradvector = new double[2][2][sisjbj[0][0].length];

		double smooth = 0.00000000001;

		/*
		 * double[] lambda= new double[aLen+bLen]; System.arraycopy(nparam, 0,
		 * lambda, 0, aLen); System.arraycopy(eparam, 0, lambda, aLen, bLen);
		 * int lamlength = lambda.length; double[] gradvector = new
		 * double[lambda.length];
		 */

		// int sentlength = deptree.length;

		ptl = inf.lookupMarginal(allVars[0]); // ptl will be the corresponding
		// factor of the variable var
		// assignment iterator method is to query a factor ptl of the variable
		// so
		double normalise = ptl.value(ptl.assignmentIterator());

		if (polarity == -1) {

		} else {
			normalise = 1 - normalise;
		}

		if (normalise == 0)
			System.out.println("normalise problem");

		for (int z = 0; z < Math.pow(2, nphrases); z++) { // loop for different
			// configurations

			String binString = Integer.toBinaryString(z);
			int binlength = binString.length();
			int[] outcome = new int[nphrases + 1];
			outcome[0] = (polarity + 1) / 2;
			for (int j = 1; j < nphrases + 1; j++) {

				if (j > binString.length())
					outcome[j] = 0;
				else {
					if (binString.charAt(binlength - j) == '0')
						outcome[j] = 0;
					else
						outcome[j] = 1;
				}

			}
			/*
			 * System.out.println(binString+"\n"); for (int k = 0; k <
			 * outcome.length;k++){ System.out.println(outcome[k]); }
			 */

			assn = new Assignment(mdl, outcome);
			double config1 = Math.exp(inf.lookupLogJoint(assn));

			outcome[0] = 1 - outcome[0];
			assn = new Assignment(mdl, outcome);
			double config2 = Math.exp(inf.lookupLogJoint(assn));
			outcome[0] = 1 - outcome[0];

			double[] siacc = new double[2];
			double[][] siqiacc = new double[2][3];
			double[][][] siqiriacc = new double[2][3][2]; // surface form
			// feature is being
			// omitted, also no
			// coarse grained
			// postag is there
			double[][] sifiacc = new double[2][46];
			double[][] sibiacc = new double[sibi.length][sibi[0].length];

			double[][] sisjacc = new double[2][2];
			double[][][] sisjrjacc = new double[2][2][2];
			double[][][][] sisjrjqjacc = new double[2][2][2][3];
			double[][][] sisjbiacc = new double[2][2][sisjbi[0][0].length];
			double[][][] sisjbjacc = new double[2][2][sisjbj[0][0].length];

			double[][] sisjaccalt = new double[2][2];
			double[][][] sisjrjaccalt = new double[2][2][2];
			double[][][][] sisjrjqjaccalt = new double[2][2][2][3];
			double[][][] sisjbiaccalt = new double[2][2][sisjbi[0][0].length];
			double[][][] sisjbjaccalt = new double[2][2][sisjbj[0][0].length];

			for (int i = 0; i < nphrases; i++) {
				siacc[outcome[i + 1]] += 1;
				siqiacc[outcome[i + 1]][priorpol[i] + 1] += 1;
				siqiriacc[outcome[i + 1]][priorpol[i] + 1][reversepol[i]] += 1;
				for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
					if (postag.get(deptree[j][3]) != null) {
						sifiacc[outcome[i + 1]][postag.get(deptree[j][3]) - 1] += 1;
					}
					if (j <= ptree[i][1]) {
						if (baseform.get(deptree[j][2]) != null) {
							sibiacc[outcome[i + 1]][baseform.get(deptree[j][2]) - 1] += 1;
						}
					}
				}

				int head;
				int headphrase = 0;
				boolean found = false;
				for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
					head = Integer.parseInt(deptree[j][5]);
					if (head != 0)
						if (phraseno[head - 1] != i) {
							headphrase = phraseno[head - 1];
							found = true;
							break;
						}
				}
				if (!found) {
					sisjacc[outcome[i + 1]][outcome[0]] += 1;
					sisjaccalt[outcome[i + 1]][1 - outcome[0]] += 1;
					for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
						if (baseform.get(deptree[j][2]) != null) {
							sisjbiacc[outcome[i + 1]][outcome[0]][baseform.get(deptree[j][2]) - 1] += 1;
							sisjbiaccalt[outcome[i + 1]][1 - outcome[0]][baseform.get(deptree[j][2]) - 1] += 1;
						}
					}
				} else {
					sisjacc[outcome[i + 1]][outcome[headphrase + 1]] += 1;
					sisjaccalt[outcome[i + 1]][outcome[headphrase + 1]] += 1;

					sisjrjacc[outcome[i + 1]][outcome[headphrase + 1]][reversepol[headphrase]] += 1;
					sisjrjaccalt[outcome[i + 1]][outcome[headphrase + 1]][reversepol[headphrase]] += 1;

					sisjrjqjacc[outcome[i + 1]][outcome[headphrase + 1]][reversepol[headphrase]][priorpol[headphrase]
							+ 1] += 1;
					sisjrjqjaccalt[outcome[i + 1]][outcome[headphrase + 1]][reversepol[headphrase]][priorpol[headphrase]
							+ 1] += 1;

					for (int j = ptree[i][0]; j <= ptree[i][1]; j++) {
						if (baseform.get(deptree[j][2]) != null) {
							sisjbiacc[outcome[i + 1]][outcome[headphrase + 1]][baseform.get(deptree[j][2]) - 1] += 1;
							sisjbiaccalt[outcome[i + 1]][outcome[headphrase + 1]][baseform.get(deptree[j][2]) - 1] += 1;
						}
					}
					for (int k = ptree[headphrase][0]; k <= ptree[headphrase][1]; k++) {
						if (baseform.get(deptree[k][2]) != null) {
							sisjbjacc[outcome[i + 1]][outcome[headphrase + 1]][baseform.get(deptree[k][2]) - 1] += 1;
							sisjbjaccalt[outcome[i + 1]][outcome[headphrase + 1]][baseform.get(deptree[k][2]) - 1] += 1;
						}
					}
				}

			}

			for (int i = 0; i < si.length; i++)
				sigradvector[i] += config1 * siacc[i] / (normalise + smooth) - config1 * siacc[i] - config2 * siacc[i];

			for (int i = 0; i < siqi.length; i++)
				for (int j = 0; j < siqi[0].length; j++)
					siqigradvector[i][j] += config1 * siqiacc[i][j] / (normalise + smooth) - config1 * siqiacc[i][j]
							- config2 * siqiacc[i][j];

			for (int i = 0; i < siqiri.length; i++)
				for (int j = 0; j < siqiri[0].length; j++)
					for (int k = 0; k < siqiri[0][0].length; k++)
						siqirigradvector[i][j][k] += config1 * siqiriacc[i][j][k] / (normalise + smooth)
						- config1 * siqiriacc[i][j][k] - config2 * siqiriacc[i][j][k];

			for (int i = 0; i < sifi.length; i++)
				for (int j = 0; j < sifi[0].length; j++)
					sifigradvector[i][j] += config1 * sifiacc[i][j] / (normalise + smooth) - config1 * sifiacc[i][j]
							- config2 * sifiacc[i][j];

			for (int i = 0; i < sibi.length; i++)
				for (int j = 0; j < sibi[0].length; j++)
					sibigradvector[i][j] += config1 * sibiacc[i][j] / (normalise + smooth) - config1 * sibiacc[i][j]
							- config2 * sibiacc[i][j];

			for (int i = 0; i < sisj.length; i++)
				for (int j = 0; j < sisj[0].length; j++)
					sisjgradvector[i][j] += config1 * sisjacc[i][j] / (normalise + smooth) - config1 * sisjacc[i][j]
							- config2 * sisjaccalt[i][j];

			for (int i = 0; i < sisjrj.length; i++)
				for (int j = 0; j < sisjrj[0].length; j++)
					for (int k = 0; k < sisjrj[0][0].length; k++)
						sisjrjgradvector[i][j][k] += config1 * sisjrjacc[i][j][k] / (normalise + smooth)
						- config1 * sisjrjacc[i][j][k] - config2 * sisjrjaccalt[i][j][k];

			for (int i = 0; i < sisjrjqj.length; i++)
				for (int j = 0; j < sisjrjqj[0].length; j++)
					for (int k = 0; k < sisjrjqj[0][0].length; k++)
						for (int l = 0; l < sisjrjqj[0][0][0].length; l++)
							sisjrjqjgradvector[i][j][k][l] += config1 * sisjrjqjacc[i][j][k][l] / (normalise + smooth)
							- config1 * sisjrjqjacc[i][j][k][l] - config2 * sisjrjqjaccalt[i][j][k][l];

			for (int i = 0; i < sisjbi.length; i++)
				for (int j = 0; j < sisjbi[0].length; j++)
					for (int k = 0; k < sisjbi[0][0].length; k++)
						sisjbigradvector[i][j][k] += config1 * sisjbiacc[i][j][k] / (normalise + smooth)
						- config1 * sisjbiacc[i][j][k] - config2 * sisjbiaccalt[i][j][k];

			for (int i = 0; i < sisjbj.length; i++)
				for (int j = 0; j < sisjbj[0].length; j++)
					for (int k = 0; k < sisjbj[0][0].length; k++)
						sisjbjgradvector[i][j][k] += config1 * sisjbjacc[i][j][k] / (normalise + smooth)
						- config1 * sisjbjacc[i][j][k] - config2 * sisjbjaccalt[i][j][k];

		}

		// last step
		for (int i = 0; i < si.length; i++)
			sigrad[i] += sigradvector[i];

		for (int i = 0; i < siqi.length; i++)
			for (int j = 0; j < siqi[0].length; j++)
				siqigrad[i][j] += siqigradvector[i][j];

		for (int i = 0; i < siqiri.length; i++)
			for (int j = 0; j < siqiri[0].length; j++)
				for (int k = 0; k < siqiri[0][0].length; k++)
					siqirigrad[i][j][k] += siqirigradvector[i][j][k];

		for (int i = 0; i < sifi.length; i++)
			for (int j = 0; j < sifi[0].length; j++)
				sifigrad[i][j] += sifigradvector[i][j];

		for (int i = 0; i < sibi.length; i++)
			for (int j = 0; j < sibi[0].length; j++)
				sibigrad[i][j] += sibigradvector[i][j];

		for (int i = 0; i < sisj.length; i++)
			for (int j = 0; j < sisj[0].length; j++)
				sisjgrad[i][j] += sisjgradvector[i][j];

		for (int i = 0; i < sisjrj.length; i++)
			for (int j = 0; j < sisjrj[0].length; j++)
				for (int k = 0; k < sisjrj[0][0].length; k++)
					sisjrjgrad[i][j][k] += sisjrjgradvector[i][j][k];

		for (int i = 0; i < sisjrjqj.length; i++)
			for (int j = 0; j < sisjrjqj[0].length; j++)
				for (int k = 0; k < sisjrjqj[0][0].length; k++)
					for (int l = 0; l < sisjrjqj[0][0][0].length; l++)
						sisjrjqjgrad[i][j][k][l] += sisjrjqjgradvector[i][j][k][l];

		for (int i = 0; i < sisjbi.length; i++)
			for (int j = 0; j < sisjbi[0].length; j++)
				for (int k = 0; k < sisjbi[0][0].length; k++)
					sisjbigrad[i][j][k] += sisjbigradvector[i][j][k];

		for (int i = 0; i < sisjbj.length; i++)
			for (int j = 0; j < sisjbj[0].length; j++)
				for (int k = 0; k < sisjbj[0][0].length; k++)
					sisjbjgrad[i][j][k] += sisjbjgradvector[i][j][k];

		/*
		 * 
		 * Factor ptl = inf.lookupMarginal (allVars[0]); //ptl will be the
		 * corresponding factor of the variable var //assignment iterator method
		 * is to query a factor ptl of the variable so System.out.println(
		 * "Marginals  "+ ptl.value(ptl.assignmentIterator ()));
		 */

		/*
		 * double[][] mytable = new double[deptree.length+1][2]; double[][]
		 * edgetable = new double[deptree.length+1][4]; // its first entry will
		 * remain as empty since root node has no parent for(int i = 0;i <
		 * deptree.length + 1;i++){ Factor ptl1 =
		 * inf.lookupMarginal(allVars[i]); mytable if(i!=0){ Factor ptl2 =
		 * inf.lookupMarginal(allvars[i], allvars[deptree[i-1][5]]); } }
		 * 
		 * nvar = deptree.length + 1; double nconfig = Math.pow(2, nvar);
		 * double[] config = new double[nconfig]; for (int i = 0; i < nconfig;
		 * i++){
		 * 
		 * }
		 */

	}

	/*
	 * public static double[] Optimisation(String[][] deptree, double[] nparam,
	 * double[] eparam, int polarity, ) {
	 * 
	 * //OPTIMISATION BEGINS }
	 */

	public static void convtoparams(double[][] params, double[] si, double[][] siqi, double[][][] siqiri,
			double[][] sifi, double[][] sibi, double[][] sisj, double[][][] sisjrj, double[][][][] sisjrjqj,
			double[][][] sisjbi, double[][][] sisjbj) {

		int count = 0;
		for (int i = 0; i < si.length; i++)
			params[count++][0] = si[i];

		for (int i = 0; i < siqi.length; i++)
			for (int j = 0; j < siqi[0].length; j++)
				params[count++][0] = siqi[i][j];

		for (int i = 0; i < siqiri.length; i++)
			for (int j = 0; j < siqiri[0].length; j++)
				for (int k = 0; k < siqiri[0][0].length; k++)
					params[count++][0] = siqiri[i][j][k];

		for (int i = 0; i < sifi.length; i++)
			for (int j = 0; j < sifi[0].length; j++)
				params[count++][0] = sifi[i][j];

		for (int i = 0; i < sibi.length; i++)
			for (int j = 0; j < sibi[0].length; j++)
				params[count++][0] = sibi[i][j];

		for (int i = 0; i < sisj.length; i++)
			for (int j = 0; j < sisj[0].length; j++)
				params[count++][0] = sisj[i][j];

		for (int i = 0; i < sisjrj.length; i++)
			for (int j = 0; j < sisjrj[0].length; j++)
				for (int k = 0; k < sisjrj[0][0].length; k++)
					params[count++][0] = sisjrj[i][j][k];

		for (int i = 0; i < sisjrjqj.length; i++)
			for (int j = 0; j < sisjrjqj[0].length; j++)
				for (int k = 0; k < sisjrjqj[0][0].length; k++)
					for (int l = 0; l < sisjrjqj[0][0][0].length; l++)
						params[count++][0] = sisjrjqj[i][j][k][l];

		for (int i = 0; i < sisjbi.length; i++)
			for (int j = 0; j < sisjbi[0].length; j++)
				for (int k = 0; k < sisjbi[0][0].length; k++)
					params[count++][0] = sisjbi[i][j][k];

		for (int i = 0; i < sisjbj.length; i++)
			for (int j = 0; j < sisjbj[0].length; j++)
				for (int k = 0; k < sisjbj[0][0].length; k++)
					params[count++][0] = sisjbj[i][j][k];
	}

	public static int phrasetree(String[][] deptree, int[] phrase) {
		int nphrases = deptree.length;
		boolean change;
		int iter = 0;
		int[] combined = new int[deptree.length - 1];
		phrase[deptree.length - 1] = deptree.length - 1;
		do {
			change = false;

			for (int i = 0; i < deptree.length - 1; i++) {
				if (combined[i] == 0) {
					boolean pass1 = false;
					boolean join = false;
					if (i != 0)
						if (combined[i - 1] == 0)
							phrase[i] = i;
					String yi = deptree[i][3];
					String yi1 = deptree[i + 1][3];
					int hi = Integer.parseInt(deptree[i][5]);
					int hi1 = Integer.parseInt(deptree[i + 1][5]);
					if ((hi == hi1 || hi - 1 == i + 1 || hi1 - 1 == i) && iter == 0)
						pass1 = true;
					if (iter != 0)
						if (hi == hi1 || hi - 1 == i + 1 || hi1 - 1 == i)
							pass1 = true;
						else {
							if (hi != 0 && hi1 != 0)
								if (phrase[hi - 1] == phrase[hi1 - 1])
									pass1 = true;
							// for further combination, check whether any word
							// in phrase has head in phrase containg i+1
							if (hi != 0)
								if (phrase[hi - 1] == phrase[i + 1])
									pass1 = true;
							if (hi1 != 0)
								if (phrase[hi1 - 1] == phrase[i])
									pass1 = true;
						}
					if (pass1) {
						if (LTH.contains(yi))
							join = true;
						else if (RTH.contains(yi1))
							join = true;
						else if (PPH.contains(yi))
							join = true;
						else if (NNH.contains(yi) && NNH.contains(yi1))
							join = true;
					}

					if (join) {
						nphrases -= 1;
						change = true;
						combined[i] = 1;
						int cont = phrase[i + 1];
						for (int j = i + 1; j < deptree.length; j++) {
							if (phrase[j] == cont && combined[j - 1] == 1)
								phrase[j] = phrase[i]; // this word is included
							// in a phrase starting
							// from word with index
							// = word number - 1
							else
								break;
						}

					}
				}
			}
			iter = iter + 1;
		} while (change);
		return nphrases;
	}

	static Map<String, Integer> priorSent;
	static Set<String> polrev;
	static Set<String> fNeg;
	static Map<String, Integer> postag;
	static Map<String, Integer> baseform;

	static Set<String> LTH;
	static Set<String> RTH;
	static Set<String> PPH;
	static Set<String> NNH;

	public static void main(String args[]) throws Exception {

		String LT[] = { "\"", "-LRB-", "CC" };
		String RT[] = { "\"", "-RRB-", "POS", ".", ",", ":" };
		String PP[] = { "IN", "RP", "TO", "DT", "PDT", "PRP", "WDT", "WP", "WP$", "WRB" };
		String NN[] = { "CD", "FW", "NN", "NNP", "NNPS", "NNS", "SYM", "JJ" };

		LTH = new HashSet<String>();
		RTH = new HashSet<String>();
		PPH = new HashSet<String>();
		NNH = new HashSet<String>();

		int q = 0;
		while (q < 3) {
			LTH.add(LT[q]);
			q += 1;
		}

		q = 0;
		while (q < 6) {
			RTH.add(RT[q]);
			q += 1;
		}

		q = 0;
		while (q < 10) {
			PPH.add(PP[q]);
			q += 1;
		}

		q = 0;
		while (q < 8) {
			NNH.add(NN[q]);
			q += 1;
		}

		String[][][] s = new String[10][][];
		deptree t = new deptree();
		// PrintStream output = null;
		PrintStream output = new PrintStream(new File("/Users/madhur/Downloads/MiniP/data/baseforms.txt"));
		BufferedReader br3 = new BufferedReader(
				new FileReader("/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.pos"));
		int count = 0;
		Set<String> basedict = new HashSet<String>();

		int actual1 = 0;
		int actual2 = 0;
		int countpos = 221;
		int countneg = 166;

		while (count != countpos) {
			count = count + 1;
			s = t.get(br3.readLine());

			for (int i = 0; i < s.length; i++) {
				if (s[i].length > 2 & s[i].length <= 16) {
					actual1 += 1;
					for (int j = 0; j < s[i].length; j++) {
						if (basedict.contains(s[i][j][2])) {
						}

						else {
							basedict.add(s[i][j][2]);
							output.println(s[i][j][2]);
						}
					}
				}
			}

		}
		BufferedReader br4 = new BufferedReader(
				new FileReader("/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.neg"));
		count = 0;

		while (count != countneg) {
			count = count + 1;
			s = t.get(br4.readLine());

			for (int i = 0; i < s.length; i++) {
				if (s[i].length > 2 & s[i].length <= 16) {
					actual2 += 1;
					for (int j = 0; j < s[i].length; j++) {
						if (basedict.contains(s[i][j][2])) {
						}

						else {
							basedict.add(s[i][j][2]);
							output.println(s[i][j][2]);
						}
					}
				}

			}

		}

		output.close();
		System.out.println(actual1 + " numbers " + actual2);

		priorSent = new HashMap<String, Integer>();
		polrev = new HashSet<String>();
		fNeg = new HashSet<String>();
		postag = new HashMap<String, Integer>();
		baseform = new HashMap<String, Integer>();

		BufferedReader br = null;
		String sCurrentLine;

		br = new BufferedReader(new FileReader(
				"/Users/madhur/Downloads/MiniP/data/prior polarity/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"));

		while ((sCurrentLine = br.readLine()) != null) {
			int pos = sCurrentLine.indexOf("word1=") + 6;
			int j = pos;
			while (sCurrentLine.charAt(j) != ' ') {
				j++;
			}
			String word = sCurrentLine.substring(pos, j);
			if (priorSent.get(word) == null) {
				int pos2 = sCurrentLine.indexOf("priorpolarity=") + 14;
				String x = sCurrentLine.substring(pos2, pos2 + 3);

				if (x.equals("pos")) {
					priorSent.put(word, 1);
				} else if (x.equals("neg")) {
					priorSent.put(word, -1);
				} else {
					priorSent.put(word, 0);
				}

			}
		}
		/*
		 * Set<String> keys = priorSent.keySet(); Iterator<String> it =
		 * keys.iterator(); while (it.hasNext()) { String word = it.next();
		 * System.out.println(word + " " + priorSent.get(word)); }
		 * 
		 */
		BufferedReader br2 = null;
		String sCurrentLine2;

		br2 = new BufferedReader(new FileReader("/Users/madhur/Downloads/MiniP/data/polarityrev.txt"));

		while ((sCurrentLine2 = br2.readLine()) != null) {

			polrev.add(sCurrentLine2);
		}

		BufferedReader brneg = null;
		brneg = new BufferedReader(new FileReader("/Users/madhur/Downloads/MiniP/data/function_negator.txt"));

		while ((sCurrentLine2 = brneg.readLine()) != null) {

			fNeg.add(sCurrentLine2);
		}
		/*
		 * Iterator<String> it2 = polrev.iterator();
		 * 
		 * while (it2.hasNext()) {
		 * 
		 * System.out.println(it2.next()); }
		 * 
		 */
		BufferedReader brpos = null;
		brpos = new BufferedReader(new FileReader("/Users/madhur/Downloads/MiniP/data/postags.txt"));
		int poslabel = 1;

		while ((sCurrentLine = brpos.readLine()) != null) {

			if (postag.get(sCurrentLine) == null) {

				postag.put(sCurrentLine, poslabel);
				poslabel += 1;
			}
		}
		brpos = new BufferedReader(new FileReader("/Users/madhur/Downloads/MiniP/data/baseforms.txt"));
		int baselabel = 1;
		while ((sCurrentLine = brpos.readLine()) != null) {

			if (baseform.get(sCurrentLine) == null) {

				baseform.put(sCurrentLine, baselabel);
				baselabel += 1;
			}
		}
		s = new String[10][][];
		System.out.println("Good to Go!");

		// TO INITIALISE THE PARAMETERS MANUALLY
		double[] si = new double[2];
		double[][] siqi = new double[2][3];
		double[][][] siqiri = new double[2][3][2]; // surface form feature is
		// being omitted, also no
		// coarse grained postag is
		// there
		double[][] sifi = new double[2][46];
		double[][] sibi = new double[2][baselabel];

		double[][] sisj = new double[2][2];
		double[][][] sisjrj = new double[2][2][2];
		double[][][][] sisjrjqj = new double[2][2][2][3];
		double[][][] sisjbi = new double[2][2][baselabel];
		double[][][] sisjbj = new double[2][2][baselabel];

		int total = 2 + 2 * 3 + 2 * 3 * 2 + 2 * 46 + 2 * baselabel + 2 * 2 + 2 * 2 * 2 + 2 * 2 * 2 * 3
				+ 2 * 2 * baselabel + 2 * 2 * baselabel;
		double[][] params = new double[total][1];
		double[][] gradold = new double[total][1];
		double[][] gradnew = new double[total][1];
		double[][] bkinv = new double[total][total];
		double[][] sk = new double[total][1];
		double[][] yk = new double[total][1];
		identity(bkinv);

		Random Randomno = new Random();

		for (int i = 0; i < siqi.length; i++)
			for (int j = 0; j < siqi[0].length; j++) {
				siqi[i][j] = -0.1 + Randomno.nextInt(21) * 0.01;
				while (siqi[i][j] == 0)
					siqi[i][j] = -0.1 + Randomno.nextInt(21) * 0.01;
			}

		siqi[0][0] = 0.9;
		siqi[1][2] = 0.9;

		for (int i = 0; i < siqiri.length; i++)
			for (int j = 0; j < siqiri[0].length; j++)
				for (int k = 0; k < siqiri[0][0].length; k++) {
					siqiri[i][j][k] = -0.1 + Randomno.nextInt(21) * 0.01;
					while (siqiri[i][j][k] == 0)
						siqiri[i][j][k] = -0.1 + Randomno.nextInt(21) * 0.01;
				}

		for (int i = 0; i < sifi.length; i++)
			for (int j = 0; j < sifi[0].length; j++) {
				sifi[i][j] = -0.1 + Randomno.nextInt(21) * 0.01;
				while (sifi[i][j] == 0)
					sifi[i][j] = -0.1 + Randomno.nextInt(21) * 0.01;
			}

		for (int i = 0; i < sibi.length; i++)
			for (int j = 0; j < sibi[0].length; j++) {
				sibi[i][j] = -0.1 + Randomno.nextInt(21) * 0.01;
				while (sibi[i][j] == 0)
					sibi[i][j] = -0.1 + Randomno.nextInt(21) * 0.01;
			}

		for (int i = 0; i < sisj.length; i++)
			for (int j = 0; j < sisj[0].length; j++) {
				sisj[i][j] = -0.1 + Randomno.nextInt(21) * 0.01;
				while (sisj[i][j] == 0)
					sisj[i][j] = -0.1 + Randomno.nextInt(21) * 0.01;
			}

		sisj[0][0] = 0.9;
		sisj[1][1] = 0.9;

		for (int i = 0; i < sisjrj.length; i++)
			for (int j = 0; j < sisjrj[0].length; j++)
				for (int k = 0; k < sisjrj[0][0].length; k++) {
					sisjrj[i][j][k] = -0.1 + Randomno.nextInt(21) * 0.01;
					while (sisjrj[i][j][k] == 0)
						sisjrj[i][j][k] = -0.1 + Randomno.nextInt(21) * 0.01;
				}

		sisjrj[0][1][1] = 0.9;
		sisjrj[1][0][1] = 0.9;

		for (int i = 0; i < sisjrjqj.length; i++)
			for (int j = 0; j < sisjrjqj[0].length; j++)
				for (int k = 0; k < sisjrjqj[0][0].length; k++)
					for (int l = 0; l < sisjrjqj[0][0][0].length; l++) {
						sisjrjqj[i][j][k][l] = -0.1 + Randomno.nextInt(21) * 0.01;
						while (sisjrjqj[i][j][k][l] == 0)
							sisjrjqj[i][j][k][l] = -0.1 + Randomno.nextInt(21) * 0.01;
					}

		sisjrjqj[1][0][1][0] = 0.9;
		sisjrjqj[0][1][1][2] = 0.9;

		for (int i = 0; i < sisjbi.length; i++)
			for (int j = 0; j < sisjbi[0].length; j++)
				for (int k = 0; k < sisjbi[0][0].length; k++) {
					sisjbi[i][j][k] = -0.1 + Randomno.nextInt(21) * 0.01;
					while (sisjbi[i][j][k] == 0)
						sisjbi[i][j][k] = -0.1 + Randomno.nextInt(21) * 0.01;
				}

		for (int i = 0; i < sisjbj.length; i++)
			for (int j = 0; j < sisjbj[0].length; j++)
				for (int k = 0; k < sisjbj[0][0].length; k++) {
					sisjbj[i][j][k] = -0.1 + Randomno.nextInt(21) * 0.01;
					while (sisjbj[i][j][k] == 0)
						sisjbj[i][j][k] = -0.1 + Randomno.nextInt(21) * 0.01;
				}

		convtoparams(params, si, siqi, siqiri, sifi, sibi, sisj, sisjrj, sisjrjqj, sisjbi, sisjbj);

		/*
		 * // TO READ PARAMETERS FROM A FILE Scanner in=new Scanner(new
		 * File("/Users/madhur/Downloads/MiniP/data/model/output" +
		 * Integer.toString(9) +".txt" ) ); double[] nparams = new double[3];
		 * double[] eparams = new double[3]; double[][] posparams = new
		 * double[46][2]; double[][] baseparams = new double[baselabel][2];
		 * double[][] baseparams1 = new double[baselabel][4]; double[][]
		 * baseparams2 = new double[baselabel][4]; for(int k = 0 ; k <
		 * nparams.length; k++){ nparams[k] = in.nextDouble(); }
		 * 
		 * for(int k = 0 ; k < eparams.length; k++){ eparams[k] =
		 * in.nextDouble(); }
		 * 
		 * for (int init1 = 0; init1 < posparams.length; init1++){ for( int
		 * init2 = 0; init2 < posparams[0].length; init2++ ){
		 * posparams[init1][init2] = in.nextDouble(); } }
		 * 
		 * for (int init1 = 0; init1 < baselabel; init1++){ for( int init2 = 0;
		 * init2 < 2; init2++ ){ baseparams[init1][init2] = in.nextDouble(); } }
		 * 
		 * for (int init1 = 0; init1 < baselabel; init1++){ for( int init2 = 0;
		 * init2 < 4; init2++ ){ baseparams1[init1][init2] = in.nextDouble(); }
		 * }
		 * 
		 * for (int init1 = 0; init1 < baselabel; init1++){ for( int init2 = 0;
		 * init2 < 4; init2++ ){ baseparams2[init1][init2] = in.nextDouble(); }
		 * }
		 * 
		 * in.close();
		 * 
		 * int total = nparams.length + eparams.length + posparams.length*2 +
		 * baseparams.length*2 + baseparams1.length*4 + baseparams2.length*4;
		 * double[][] params = new double[total][1]; double[][] gradold = new
		 * double[total][1]; double[][] gradnew = new double[total][1];
		 * double[][] bkinv = new double[total][total]; double[][] sk = new
		 * double[total][1]; double[][] yk = new double[total][1];
		 * identity(bkinv); int gradcount = 6; params[0][0] = nparams[0] ;
		 * params[1][0] = nparams[1] ; params[2][0] = nparams[2] ; params[3][0]
		 * = eparams[0] ; params[4][0] = eparams[1] ; params[5][0] = eparams[2]
		 * ;
		 * 
		 */

		System.out.println("Better");
		int iter = 0;

		while (iter < 20) {
			// double eta = 1.0/(actual1 + actual2);
			double eta = 0.05;
			System.gc();

			double[] sigrad = new double[2];
			double[][] siqigrad = new double[2][3];
			double[][][] siqirigrad = new double[2][3][2]; // surface form
			// feature is being
			// omitted, also no
			// coarse grained
			// postag is there
			double[][] sifigrad = new double[2][46];
			double[][] sibigrad = new double[2][baselabel];

			double[][] sisjgrad = new double[2][2];
			double[][][] sisjrjgrad = new double[2][2][2];
			double[][][][] sisjrjqjgrad = new double[2][2][2][3];
			double[][][] sisjbigrad = new double[2][2][baselabel];
			double[][][] sisjbjgrad = new double[2][2][baselabel];

			// br3 = new BufferedReader(new
			// FileReader("/Users/madhur/Downloads/MiniP/data/synthetic.txt"));
			// br3 = new BufferedReader(new
			// FileReader("/Users/madhur/Downloads/MiniP/data/possynthetic.txt"));
			br3.close();
			br3 = new BufferedReader(
					new FileReader("/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.pos"));
			count = 0;
			while (count != countpos) {
				count = count + 1;
				s = t.get(br3.readLine());

				for (int i = 0; i < s.length; i++) {
					if (s[i].length > 2 & s[i].length <= 16) { // si is
						// dependency
						// tree, phrase
						// is an array
						// to act as a
						// dependency
						// structure
						int[] phrase = new int[s[i].length];
						int nphrases;
						nphrases = phrasetree(s[i], phrase);
						createFactorGraph(s[i], phrase, nphrases, si, siqi, siqiri, sifi, sibi, sisj, sisjrj, sisjrjqj,
								sisjbi, sisjbj, sigrad, siqigrad, siqirigrad, sifigrad, sibigrad, sisjgrad, sisjrjgrad,
								sisjrjqjgrad, sisjbigrad, sisjbjgrad, 1);
						// System.out.println(s[i].length);
					}
				}

			}
			System.out.println("Halfway There");
			br4.close();
			br4 = new BufferedReader(
					new FileReader("/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.neg"));
			count = 0;
			// br4 = new BufferedReader(new
			// FileReader("/Users/madhur/Downloads/MiniP/data/negsynthetic.txt"));
			while (count != countneg) {
				count = count + 1;
				s = t.get(br4.readLine());

				for (int i = 0; i < s.length; i++) {
					if (s[i].length > 2 & s[i].length <= 16) {
						int[] phrase = new int[s[i].length];
						int nphrases;
						nphrases = phrasetree(s[i], phrase);
						createFactorGraph(s[i], phrase, nphrases, si, siqi, siqiri, sifi, sibi, sisj, sisjrj, sisjrjqj,
								sisjbi, sisjbj, sigrad, siqigrad, siqirigrad, sifigrad, sibigrad, sisjgrad, sisjrjgrad,
								sisjrjqjgrad, sisjbigrad, sisjbjgrad, -1);
						// System.out.println(s[i].length);
					}

				}

			}
			System.gc();
			// double eta = 1/( count );

			double sigma = 2;
			double sigmasq = sigma * sigma;
			int gradcount = 0;

			for (int i = 0; i < si.length; i++) {
				sigrad[i] -= si[i] / sigmasq;
				if (iter == 0)
					gradnew[gradcount][0] = sigrad[i];
				else {
					gradold[gradcount][0] = gradnew[gradcount][0];
					gradnew[gradcount][0] = sigrad[i];
				}
				gradcount++;
			}

			for (int i = 0; i < siqi.length; i++)
				for (int j = 0; j < siqi[0].length; j++) {
					siqigrad[i][j] -= siqi[i][j] / sigmasq;
					if (iter == 0)
						gradnew[gradcount][0] = siqigrad[i][j];
					else {
						gradold[gradcount][0] = gradnew[gradcount][0];
						gradnew[gradcount][0] = siqigrad[i][j];
					}
					gradcount++;
				}

			for (int i = 0; i < siqiri.length; i++)
				for (int j = 0; j < siqiri[0].length; j++)
					for (int k = 0; k < siqiri[0][0].length; k++) {
						siqirigrad[i][j][k] -= siqiri[i][j][k] / sigmasq;
						if (iter == 0)
							gradnew[gradcount][0] = siqirigrad[i][j][k];
						else {
							gradold[gradcount][0] = gradnew[gradcount][0];
							gradnew[gradcount][0] = siqirigrad[i][j][k];
						}
						gradcount++;
					}

			for (int i = 0; i < sifi.length; i++)
				for (int j = 0; j < sifi[0].length; j++) {
					sifigrad[i][j] -= sifi[i][j] / sigmasq;
					if (iter == 0)
						gradnew[gradcount][0] = sifigrad[i][j];
					else {
						gradold[gradcount][0] = gradnew[gradcount][0];
						gradnew[gradcount][0] = sifigrad[i][j];
					}
					gradcount++;
				}

			for (int i = 0; i < sibi.length; i++)
				for (int j = 0; j < sibi[0].length; j++) {
					sibigrad[i][j] -= sibi[i][j] / sigmasq;
					if (iter == 0)
						gradnew[gradcount][0] = sibigrad[i][j];
					else {
						gradold[gradcount][0] = gradnew[gradcount][0];
						gradnew[gradcount][0] = sibigrad[i][j];
					}
					gradcount++;
				}

			for (int i = 0; i < sisj.length; i++)
				for (int j = 0; j < sisj[0].length; j++) {
					sisjgrad[i][j] -= sisj[i][j] / sigmasq;
					if (iter == 0)
						gradnew[gradcount][0] = sisjgrad[i][j];
					else {
						gradold[gradcount][0] = gradnew[gradcount][0];
						gradnew[gradcount][0] = sisjgrad[i][j];
					}
					gradcount++;
				}

			for (int i = 0; i < sisjrj.length; i++)
				for (int j = 0; j < sisjrj[0].length; j++)
					for (int k = 0; k < sisjrj[0][0].length; k++) {
						sisjrjgrad[i][j][k] -= sisjrj[i][j][k] / sigmasq;
						if (iter == 0)
							gradnew[gradcount][0] = sisjrjgrad[i][j][k];
						else {
							gradold[gradcount][0] = gradnew[gradcount][0];
							gradnew[gradcount][0] = sisjrjgrad[i][j][k];
						}
						gradcount++;
					}

			for (int i = 0; i < sisjrjqj.length; i++)
				for (int j = 0; j < sisjrjqj[0].length; j++)
					for (int k = 0; k < sisjrjqj[0][0].length; k++)
						for (int l = 0; l < sisjrjqj[0][0][0].length; l++) {
							sisjrjqjgrad[i][j][k][l] -= sisjrjqj[i][j][k][l] / sigmasq;
							if (iter == 0)
								gradnew[gradcount][0] = sisjrjqjgrad[i][j][k][l];
							else {
								gradold[gradcount][0] = gradnew[gradcount][0];
								gradnew[gradcount][0] = sisjrjqj[i][j][k][l];
							}
							gradcount++;
						}

			for (int i = 0; i < sisjbi.length; i++)
				for (int j = 0; j < sisjbi[0].length; j++)
					for (int k = 0; k < sisjbi[0][0].length; k++) {
						sisjbigrad[i][j][k] -= sisjbi[i][j][k] / sigmasq;
						if (iter == 0)
							gradnew[gradcount][0] = sisjbigrad[i][j][k];
						else {
							gradold[gradcount][0] = gradnew[gradcount][0];
							gradnew[gradcount][0] = sisjbigrad[i][j][k];
						}
						gradcount++;
					}

			for (int i = 0; i < sisjbj.length; i++)
				for (int j = 0; j < sisjbj[0].length; j++)
					for (int k = 0; k < sisjbj[0][0].length; k++) {
						sisjbjgrad[i][j][k] -= sisjbj[i][j][k] / sigmasq;
						if (iter == 0)
							gradnew[gradcount][0] = sisjbjgrad[i][j][k];
						else {
							gradold[gradcount][0] = gradnew[gradcount][0];
							gradnew[gradcount][0] = sisjbjgrad[i][j][k];
						}
						gradcount++;
					}

			norm(gradnew);

			System.gc();
			if (iter == 0) {

				System.out.println("hello hello hello");
				double[][] dummy = new double[gradnew.length][gradnew[0].length];
				equate(dummy, gradnew);
				double[][] single = new double[1][1];
				double[][] pt = new double[1][total];
				transpose(dummy, pt);
				multiply(pt, gradnew, single);
				double m = single[0][0];
				double t2 = 0.1 * m;

				double[][] paramsnew = new double[total][1];

				double[] sinew = new double[2];
				double[][] siqinew = new double[2][3];
				double[][][] siqirinew = new double[2][3][2]; // surface form
				// feature is
				// being
				// omitted, also
				// no coarse
				// grained
				// postag is
				// there
				double[][] sifinew = new double[2][46];
				double[][] sibinew = new double[2][baselabel];

				double[][] sisjnew = new double[2][2];
				double[][][] sisjrjnew = new double[2][2][2];
				double[][][][] sisjrjqjnew = new double[2][2][2][3];
				double[][][] sisjbinew = new double[2][2][baselabel];
				double[][][] sisjbjnew = new double[2][2][baselabel];

				double first = 0;
				double second = 0;
				boolean firstiter = true;

				while ((second - first <= 0) || firstiter == true) {
					System.out.println("eta " + eta);

					second = 0;
					equate(dummy, gradnew);
					scalar(dummy, eta);
					addition(params, dummy, paramsnew);
					converter(sinew, siqinew, siqirinew, sifinew, sibinew, sisjnew, sisjrjnew, sisjrjqjnew, sisjbinew,
							sisjbjnew, paramsnew);
					boolean firstprior = true;

					br3.close();
					br3 = new BufferedReader(new FileReader(
							"/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.pos"));
					count = 0;
					while (count != countpos) {
						count = count + 1;
						s = t.get(br3.readLine());

						for (int i = 0; i < s.length; i++) {
							if (s[i].length > 2 & s[i].length <= 16) {
								int[] phrase = new int[s[i].length];
								int nphrases;
								nphrases = phrasetree(s[i], phrase);
								if (firstiter)
									first += marginalise(s[i], phrase, nphrases, si, siqi, siqiri, sifi, sibi, sisj,
											sisjrj, sisjrjqj, sisjbi, sisjbj, params, 1, firstprior);
								second += marginalise(s[i], phrase, nphrases, sinew, siqinew, siqirinew, sifinew,
										sibinew, sisjnew, sisjrjnew, sisjrjqjnew, sisjbinew, sisjbjnew, paramsnew, 1,
										firstprior);
								if (firstprior)
									firstprior = false;
							}
						}

					}

					br4.close();
					br4 = new BufferedReader(new FileReader(
							"/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.neg"));
					count = 0;
					// br4 = new BufferedReader(new
					// FileReader("/Users/madhur/Downloads/MiniP/data/negsynthetic.txt"));
					while (count != countneg) {
						count = count + 1;
						s = t.get(br4.readLine());

						for (int i = 0; i < s.length; i++) {
							if (s[i].length > 2 & s[i].length <= 16) {
								int[] phrase = new int[s[i].length];
								int nphrases;
								nphrases = phrasetree(s[i], phrase);
								if (firstiter)
									first += marginalise(s[i], phrase, nphrases, si, siqi, siqiri, sifi, sibi, sisj,
											sisjrj, sisjrjqj, sisjbi, sisjbj, params, -1, firstprior);

								second += marginalise(s[i], phrase, nphrases, sinew, siqinew, siqirinew, sifinew,
										sibinew, sisjnew, sisjrjnew, sisjrjqjnew, sisjbinew, sisjbjnew, paramsnew, -1,
										firstprior);
							}

						}

					}
					if ((second - first) <= 0) {
						eta = eta * 0.5;
					}
					firstiter = false;
				}

				equate(sk, dummy);
				equate(params, paramsnew);

			}

			else {
				System.out.println("hello hello hello");
				double[][] dummy = new double[gradold.length][gradold[0].length];
				equate(dummy, gradold);
				scalar(dummy, -1);
				addition(gradnew, dummy, yk);
				dummy = new double[1][1];
				System.gc();
				double[][] skt = new double[sk[0].length][sk.length];
				transpose(sk, skt);
				multiply(skt, yk, dummy); // dummy = skt*yk

				double[][] ykt = new double[yk[0].length][yk.length];
				System.gc();
				transpose(yk, ykt);
				double[][] dummy2 = new double[1][total];
				multiply(ykt, bkinv, dummy2);
				double[][] dummy3 = new double[1][1];
				multiply(dummy2, yk, dummy3);
				double coef1 = (dummy[0][0] + dummy3[0][0]) / (dummy[0][0] * dummy[0][0]);

				System.gc();
				double[][] skskt = new double[total][total];
				multiply(sk, skt, skskt);
				scalar(skskt, coef1); // now skskt is being modified
				System.gc();

				dummy2 = new double[total][1];
				multiply(bkinv, yk, dummy2);
				System.gc();
				double[][] dummy4 = new double[total][total];
				double[][] dummy5 = new double[total][total];
				multiply(dummy2, skt, dummy4);
				transpose(dummy4, dummy5);
				addition(dummy4, dummy5, dummy4);
				scalar(dummy4, -1 * (1 / dummy[0][0]));
				// bkinv, skskt, dummy4
				addition(bkinv, skskt, bkinv);
				addition(bkinv, dummy4, bkinv);
				System.gc();

				multiply(bkinv, gradnew, dummy2); // dummy2 now is pk

				double[][] single = new double[1][1];
				double[][] pt = new double[1][total];
				transpose(dummy2, pt);
				multiply(pt, gradnew, single);
				double m = single[0][0];
				double t2 = 0.1 * m;

				double[][] paramsnew = new double[total][1];

				double[] sinew = new double[2];
				double[][] siqinew = new double[2][3];
				double[][][] siqirinew = new double[2][3][2]; // surface form
				// feature is
				// being
				// omitted, also
				// no coarse
				// grained
				// postag is
				// there
				double[][] sifinew = new double[2][46];
				double[][] sibinew = new double[2][baselabel];

				double[][] sisjnew = new double[2][2];
				double[][][] sisjrjnew = new double[2][2][2];
				double[][][][] sisjrjqjnew = new double[2][2][2][3];
				double[][][] sisjbinew = new double[2][2][baselabel];
				double[][][] sisjbjnew = new double[2][2][baselabel];

				double first = 0;
				double second = 0;
				boolean firstiter = true;
				dummy = new double[total][1];
				while ((second - first <= 0) || firstiter == true) {

					System.out.println("eta " + eta);

					second = 0;

					equate(dummy, dummy2);
					scalar(dummy, eta);
					addition(params, dummy, paramsnew);
					converter(sinew, siqinew, siqirinew, sifinew, sibinew, sisjnew, sisjrjnew, sisjrjqjnew, sisjbinew,
							sisjbjnew, paramsnew);
					boolean firstprior = true;

					br3.close();
					br3 = new BufferedReader(new FileReader(
							"/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.pos"));
					count = 0;
					while (count != countpos) {
						count = count + 1;
						s = t.get(br3.readLine());

						for (int i = 0; i < s.length; i++) {
							if (s[i].length > 2 & s[i].length <= 16) {
								int[] phrase = new int[s[i].length];
								int nphrases;
								nphrases = phrasetree(s[i], phrase);
								if (firstiter)
									first += marginalise(s[i], phrase, nphrases, si, siqi, siqiri, sifi, sibi, sisj,
											sisjrj, sisjrjqj, sisjbi, sisjbj, params, 1, firstprior);

								second += marginalise(s[i], phrase, nphrases, sinew, siqinew, siqirinew, sifinew,
										sibinew, sisjnew, sisjrjnew, sisjrjqjnew, sisjbinew, sisjbjnew, paramsnew, 1,
										firstprior);
								if (firstprior)
									firstprior = false;
							}
						}

					}

					br4.close();
					br4 = new BufferedReader(new FileReader(
							"/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.neg"));
					count = 0;
					// br4 = new BufferedReader(new
					// FileReader("/Users/madhur/Downloads/MiniP/data/negsynthetic.txt"));
					while (count != countneg) {
						count = count + 1;
						s = t.get(br4.readLine());

						for (int i = 0; i < s.length; i++) {
							if (s[i].length > 2 & s[i].length <= 16) {
								int[] phrase = new int[s[i].length];
								int nphrases;
								nphrases = phrasetree(s[i], phrase);
								if (firstiter)
									first += marginalise(s[i], phrase, nphrases, si, siqi, siqiri, sifi, sibi, sisj,
											sisjrj, sisjrjqj, sisjbi, sisjbj, params, -1, firstprior);
								// System.out.println("first " + first);
								second += marginalise(s[i], phrase, nphrases, sinew, siqinew, siqirinew, sifinew,
										sibinew, sisjnew, sisjrjnew, sisjrjqjnew, sisjbinew, sisjbjnew, paramsnew, -1,
										firstprior);
							}

						}

					}

					if (second - first <= 0) {
						if (second - first < 0) {
							System.out.println("ERROR");
						}
						eta = eta * 0.5;
					}
					firstiter = false;
				}

				equate(sk, dummy);
				equate(params, paramsnew);

			}

			/*
			 * for(int k = 0; k<nparams.length; k++){
			 * System.out.println(nparams[k] + "\t" + eparams[k] + "\t" +
			 * egrad[0]); }
			 */
			System.gc();
			gradcount = 0;
			iter = iter + 1;
			System.out.println("iteration " + iter + " completed");

			output = new PrintStream(
					new File("/Users/madhur/Downloads/MiniP/data/model/output" + Integer.toString(iter) + ".txt"));

			for (int i = 0; i < si.length; i++) {
				si[i] = params[gradcount++][0];
				output.println(si[i]);
			}

			for (int i = 0; i < siqi.length; i++)
				for (int j = 0; j < siqi[0].length; j++) {
					siqi[i][j] = params[gradcount++][0];
					output.println(siqi[i][j]);
				}

			for (int i = 0; i < siqiri.length; i++)
				for (int j = 0; j < siqiri[0].length; j++)
					for (int k = 0; k < siqiri[0][0].length; k++) {
						siqiri[i][j][k] = params[gradcount++][0];
						output.println(siqiri[i][j][k]);
					}

			for (int i = 0; i < sifi.length; i++)
				for (int j = 0; j < sifi[0].length; j++) {
					sifi[i][j] = params[gradcount++][0];
					output.println(sifi[i][j]);
				}

			for (int i = 0; i < sibi.length; i++)
				for (int j = 0; j < sibi[0].length; j++) {
					sibi[i][j] = params[gradcount++][0];
					output.println(sibi[i][j]);
				}

			for (int i = 0; i < sisj.length; i++)
				for (int j = 0; j < sisj[0].length; j++) {
					sisj[i][j] = params[gradcount++][0];
					output.println(sisj[i][j]);
				}

			for (int i = 0; i < sisjrj.length; i++)
				for (int j = 0; j < sisjrj[0].length; j++)
					for (int k = 0; k < sisjrj[0][0].length; k++) {
						sisjrj[i][j][k] = params[gradcount++][0];
						output.println(sisjrj[i][j][k]);
					}

			for (int i = 0; i < sisjrjqj.length; i++)
				for (int j = 0; j < sisjrjqj[0].length; j++)
					for (int k = 0; k < sisjrjqj[0][0].length; k++)
						for (int l = 0; l < sisjrjqj[0][0][0].length; l++) {
							sisjrjqj[i][j][k][l] = params[gradcount++][0];
							output.println(sisjrjqj[i][j][k][l]);
						}

			for (int i = 0; i < sisjbi.length; i++)
				for (int j = 0; j < sisjbi[0].length; j++)
					for (int k = 0; k < sisjbi[0][0].length; k++) {
						sisjbi[i][j][k] = params[gradcount++][0];
						output.println(sisjbi[i][j][k]);
					}

			for (int i = 0; i < sisjbj.length; i++)
				for (int j = 0; j < sisjbj[0].length; j++)
					for (int k = 0; k < sisjbj[0][0].length; k++) {
						sisjbj[i][j][k] = params[gradcount++][0];
						output.println(sisjbj[i][j][k]);
					}

			output.close();
			System.gc();
			System.out.println("eta " + eta);
		}

	}

}

// createFactorGraph(sentence, nparam, eparam, ngrad, egrad, polarity, sigma,
// last sentence or not)

// incorporate all the features, decipher the meaning of the remaining features
// i/o sentences form the database and polarity reversal dictionary and prior
// sentiment
// parameter initialisation
// create flow for all the sentences
// do it using the phrases
// are lambdas dependent on the word????
// L-BFGS Optimisation
// optimisation to be carried out in how many iterations

// MODIFICATIONS
// add a smoothing factor
// keep others fixed
// change the initialisation
// positive and negative single words, si and qi 1, rest 0, si and sj anything,
// fix others but si and sj

// correct some of the existing features, add more features, train on a data and
// test on a data

// "." is being removed

// for 100 , for +ve - 221, for -ve - 166, >2 and <=16
// getting positively skewed
// do a histogram analysis

// parameter initialisation, remove the priors, sentence length analysis,
// experients, new model and comparisons.