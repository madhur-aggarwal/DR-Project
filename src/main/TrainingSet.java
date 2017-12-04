package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.StringReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
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
import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.FactorGraph;
import edu.umass.cs.mallet.grmm.types.HashVarSet;
import edu.umass.cs.mallet.grmm.types.TableFactor;
import edu.umass.cs.mallet.grmm.types.VarSet;
import edu.umass.cs.mallet.grmm.types.Variable;

public class TrainingSet {
	final String language = AbstractReader.LANG_EN, modelType = "general-en";

	AbstractTokenizer tokenizer;
	AbstractComponent tagger, parser, identifier, classifier, labeler;

	public TrainingSet() throws Exception {
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

	public static int edgefeature0(int si, int sj) {
		return si * sj;
	}

	public static int isequal(int one, int two) {
		if (one == two)
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

	public static double sentdet(String[][] deptree, int[] phrase, int nphrases, double[] si, double[][] siqi,
			double[][][] siqiri, double[][] sifi, double[][] sibi, double[][] sisj, double[][][] sisjrj,
			double[][][][] sisjrjqj, double[][][] sisjbi, double[][][] sisjbj) {

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

		ptl = inf.lookupMarginal(allVars[0]); // ptl will be the corresponding
		// factor of the variable var
		// assignment iterator method is to query a factor ptl of the variable
		// so
		double normalise = ptl.value(ptl.assignmentIterator());
		return normalise;

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

		/*
		 * double[] lambda= new double[aLen+bLen]; System.arraycopy(nparam, 0,
		 * lambda, 0, aLen); System.arraycopy(eparam, 0, lambda, aLen, bLen);
		 * int lamlength = lambda.length; double[] gradvector = new
		 * double[lambda.length];
		 */

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

		priorSent = new HashMap<String, Integer>();
		polrev = new HashSet<String>();
		postag = new HashMap<String, Integer>();
		fNeg = new HashSet<String>();
		baseform = new HashMap<String, Integer>();

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

		deptree t = new deptree();
		String[][][] s = new String[10][][];

		System.out.println("Good to Go!");
		double third = -1.33226762955 * Math.pow(10, -15);

		/*
		 * TO GIVE PARAMETERS MANUALLY double[] nparams = {0,2,0.0975279};
		 * double[] eparams = {2.028985,3.0,0.0975279}; double[][] posparams =
		 * new double[46][2]; for (int init1 = 0; init1 < 46; init1++){ for( int
		 * init2 = 0; init2 < 2; init2++ ){ posparams[init1][init2] = 0.1; } }
		 */
		int iter = 5; // iter is a number from 0 to number of iterations -1.
		Scanner in = new Scanner(
				new File("/Users/madhur/Downloads/MiniP/data/model/output" + Integer.toString(iter) + ".txt"));

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

		for (int i = 0; i < si.length; i++) {
			si[i] = in.nextDouble();

		}

		for (int i = 0; i < siqi.length; i++)
			for (int j = 0; j < siqi[0].length; j++) {
				siqi[i][j] = in.nextDouble();
			}

		for (int i = 0; i < siqiri.length; i++)
			for (int j = 0; j < siqiri[0].length; j++)
				for (int k = 0; k < siqiri[0][0].length; k++) {
					siqiri[i][j][k] = in.nextDouble();
				}

		for (int i = 0; i < sifi.length; i++)
			for (int j = 0; j < sifi[0].length; j++) {
				sifi[i][j] = in.nextDouble();
			}

		for (int i = 0; i < sibi.length; i++)
			for (int j = 0; j < sibi[0].length; j++) {
				sibi[i][j] = in.nextDouble();

			}

		for (int i = 0; i < sisj.length; i++)
			for (int j = 0; j < sisj[0].length; j++) {
				sisj[i][j] = in.nextDouble();

			}

		for (int i = 0; i < sisjrj.length; i++)
			for (int j = 0; j < sisjrj[0].length; j++)
				for (int k = 0; k < sisjrj[0][0].length; k++) {
					sisjrj[i][j][k] = in.nextDouble();

				}

		for (int i = 0; i < sisjrjqj.length; i++)
			for (int j = 0; j < sisjrjqj[0].length; j++)
				for (int k = 0; k < sisjrjqj[0][0].length; k++)
					for (int l = 0; l < sisjrjqj[0][0][0].length; l++) {
						sisjrjqj[i][j][k][l] = in.nextDouble();

					}

		for (int i = 0; i < sisjbi.length; i++)
			for (int j = 0; j < sisjbi[0].length; j++)
				for (int k = 0; k < sisjbi[0][0].length; k++) {
					sisjbi[i][j][k] = in.nextDouble();

				}

		for (int i = 0; i < sisjbj.length; i++)
			for (int j = 0; j < sisjbj[0].length; j++)
				for (int k = 0; k < sisjbj[0][0].length; k++) {
					sisjbj[i][j][k] = in.nextDouble();

				}
		in.close();

		// while iterating optimisation initialise ngrad and egrad to zero
		// again, then initialise old params vector to updated new values and
		// keep
		// on training using the updated parameters to get new gradient values
		// to get new parameters and find difference with old and stop when <
		// eps

		double[][] pos = new double[10000][3];
		double[][] neg = new double[10000][3];
		double marginal;
		int poscorrect = 0;
		int negcorrect = 0;
		int poswrong = 0;
		int negwrong = 0;

		BufferedReader br3 = null;

		br3 = new BufferedReader(
				new FileReader("/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.pos"));

		int lower = 1;
		int upper = 100;
		int count = 1;
		System.out.println("positive");

		while (count <= upper) {

			s = t.get(br3.readLine());

			if (count >= lower) {
				for (int i = 0; i < s.length; i++) { // only first phrase of
					// large sentence.
					if (s[i].length > 2 && s[i].length < 17) {

						int[] phrase = new int[s[i].length];
						int nphrases;
						nphrases = phrasetree(s[i], phrase);

						marginal = sentdet(s[i], phrase, nphrases, si, siqi, siqiri, sifi, sibi, sisj, sisjrj, sisjrjqj,
								sisjbi, sisjbj);
						if (marginal > 0.50) {
							pos[count - lower][0] = count;
							pos[count - lower][1] = 0;
							pos[count - lower][2] = 0;
							System.out.println(pos[count - lower][0] + " " + pos[count - lower][1] + " "
									+ pos[count - lower][2] + " " + marginal);
							poswrong++;
						} else if (marginal > 0.50) {
							pos[count - lower][0] = count;
							pos[count - lower][1] = 1;
							pos[count - lower][2] = 1;
							System.out.println(pos[count - lower][0] + " " + pos[count - lower][1] + " "
									+ pos[count - lower][2] + " " + marginal);
							// poscorrect += 1;
						} else {
							pos[count - lower][0] = count;
							pos[count - lower][1] = 1;
							pos[count - lower][2] = 1;
							System.out.println(pos[count - lower][0] + " " + pos[count - lower][1] + " "
									+ pos[count - lower][2] + " " + marginal);
							poscorrect += 1;
						}
						count = count + 1;
					}
				}
			}

		}

		BufferedReader br4 = null;
		br4 = new BufferedReader(
				new FileReader("/Users/madhur/Downloads/MiniP/data/version 1.0/rt-polaritydata/rt-polarity.neg"));

		lower = 1;
		upper = 100;
		count = 1;

		System.out.println("negative");
		while (count <= upper) {

			s = t.get(br4.readLine());

			if (count >= lower) {
				for (int i = 0; i < s.length; i++) { // only first phrase of
					// large sentence.
					if (s[i].length > 2 && s[i].length < 17) {

						int[] phrase = new int[s[i].length];
						int nphrases;
						nphrases = phrasetree(s[i], phrase);

						marginal = sentdet(s[i], phrase, nphrases, si, siqi, siqiri, sifi, sibi, sisj, sisjrj, sisjrjqj,
								sisjbi, sisjbj);
						if (marginal > 0.50) {
							neg[count - lower][0] = count;
							neg[count - lower][1] = 0;
							neg[count - lower][2] = 1;
							System.out.println(neg[count - lower][0] + " " + neg[count - lower][1] + " "
									+ neg[count - lower][2] + " " + marginal);
							negcorrect += 1;
						} else if (marginal > 0.50) {
							neg[count - lower][0] = count;
							neg[count - lower][1] = 1;
							neg[count - lower][2] = 0;
							System.out.println(neg[count - lower][0] + " " + neg[count - lower][1] + " "
									+ neg[count - lower][2] + " " + marginal);
						} else {
							neg[count - lower][0] = count;
							neg[count - lower][1] = 1;
							neg[count - lower][2] = 0;
							System.out.println(neg[count - lower][0] + " " + neg[count - lower][1] + " "
									+ neg[count - lower][2] + " " + marginal);
							negwrong++;
						}
						count = count + 1;
					}

				}
			}

		}
		System.out.println(poscorrect);
		System.out.println(poswrong);
		System.out.println(negcorrect);
		System.out.println(negwrong);

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
// deptree4 for single sentences to be given as an input, deptree5 complete
// training pipeline, deptreer to play with no. of sentences their lengths and
// deptreet for testing