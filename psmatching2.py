import os.path as _osp
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')

class PSMatch(object):
    def __init__(self, path, control, covariants, dependents=None):
        '''
        Parameters
        ----------
        path : string
            The file path of the data; assumed to be in .csv format.
        control : string
            The control variable name of the model
        covariant : list of string
            A list containing the covariant names of the model
        dependent: list of string
            A list containing the dependent variable names of the model
        '''
        self.path = path
        self.control = control
        self.covariants = covariants
        self.dependents = dependents
        df = pd.read_csv(self.path)
        self.df = df
        self.matche_id = None
        self.matched_data = None
        self.weighted_df = None


    def calculate_proprnsity_scores(self):
        '''
        Utilizes a logistic regression framework to calculate propensity scores based on a specified model.
        Results
        -------
        A Pandas DataFrame containing raw data and propensity scores.
        '''

        import statsmodels.api as sm
        model = self.control + " ~ "
        for cov in self.covariants:
            model += (cov + " + ")
        model = model[:-3]
        glm_binom = sm.formula.glm(formula=model, data=self.df, family=sm.families.Binomial())
        propensity_scores = glm_binom.fit()
        self.df["PROPENSITY"] = propensity_scores.fittedvalues
        print("\nCalculating propensity scores Done!")


    def match(self, caliper=None, replace=False, k=1):
        '''
        Performs propensity score matching.
        Parameters
        ----------
        df : Pandas DataFrame
            the attribute returned by the prepare_data() function
        capliper: float
            value between 0 and 1, the maximum difference in scores allowed during pairing, value between 0 and 1
        replace: bool
            if not allow duplicate pairing
        k：
            how many pairs of control group data are generated for each experimental group
        Returns
        -------
        matches : Pandas DataFrame
            the Match object attribute describing which control IDs are matched
            to a particular treatment case.
        matched_data: Pandas DataFrame
            the Match object attribute containing the raw data for only treatment
            cases and their matched controls.
        '''

        # Assert that the Match object has a df attribute
        if not hasattr(self, 'df'):
            raise AttributeError("%s does not have a 'df' attribute." % (self))

        # Assign treatment group membership
        groups = self.df[self.control]
        propensity = self.df.PROPENSITY
        groups = groups == groups.unique()[1]
        n = len(groups)
        n1 = groups[groups == 1].sum()
        n2 = n - n1
        g1, g2 = propensity[groups == 1], propensity[groups == 0]
        if_change = False
        if n1 > n2:
            n1, n2, g1, g2 = n2, n1, g2, g1
            if_change = True

        # Randomly permute the treatment case IDs
        if if_change:
            m_order = list(np.random.permutation(groups[groups == 0].index))
        else:

            m_order = list(np.random.permutation(groups[groups == 1].index))
        matches = {}

        # Match treatment cases to controls based on propensity score differences
        print("\nMatching [" + str(k) + "] controls to each case ... ", end=" ")
        for m in m_order:
            # Calculate all propensity score differences
            dist = abs(g1[m] - g2)
            array = np.array(dist)
            # Choose the k smallest differences
            k_smallest = np.partition(array, k)[:k].tolist()
            if caliper:
                caliper = float(caliper)
                keep_diffs = [i for i in k_smallest if i <= caliper]
                keep_ids = np.array(dist[dist.isin(keep_diffs)].index)
            else:
                keep_ids = np.array(dist[dist.isin(k_smallest)].index)
            # Break ties via random choice, if ties are present
            if len(keep_ids) > k:
                matches[m] = list(np.random.choice(keep_ids, k, replace=False))
                if replace == False:
                    g2 = g2.drop(matches[m])
            elif len(keep_ids) < k:
                pass
            else:
                matches[m] = keep_ids.tolist()
                if replace == False:
                    g2 = g2.drop(matches[m])

        # Prettify the results by consolidating into a DataFrame
        matches = pd.DataFrame.from_dict(matches, orient="index")
        matches = matches.reset_index()
        column_names = {}
        column_names["index"] = "CASE_ID"
        for i in range(k):
            column_names[i] = str("CONTROL_MATCH_" + str(i + 1))
        matches = matches.rename(columns=column_names)

        # Extract data only for treated cases and matched controls
        master_list = []
        master_list.append(matches[matches.columns[0]].tolist())
        for i in range(1, matches.shape[1]):
            master_list.append(matches[matches.columns[i]].tolist())
        master_ids = [item for sublist in master_list for item in sublist]
        matched_data = self.df[self.df.index.isin(master_ids)]
        print("DONE!")

        # Assign the matches and matched_data attributes to the Match object
        self.matche_id = matches
        self.matched_data = matched_data


    def weighted_process(self,method=None):
        '''
        Performs weight process
        Parameters
        ----------
        df : Pandas DataFrame
            the attribute returned by the match() function
        method: str
            Weight processing method, including 'IPTW' 'SMRW-T' 'SMRW-C' 'OW' 'IPTW-P'
        '''

        # 检查变量是否齐全
        pd.set_option('mode.chained_assignment', None)
        if not hasattr(self, 'df'):
            raise AttributeError("%s does not have a 'df' attribute." % (self))
        if not hasattr(self, 'matched_data'):
            raise AttributeError("%s does not have a 'matched_data' attribute." % (self))

        # IPTM
        if method == "IPTW":
            print("\nWeighted processing ...", end=" ")
            # 从matched_data浅拷贝得来weighted_df
            matched_data_len = self.matched_data.shape[0]
            self.weighted_df = self.matched_data.iloc[0:matched_data_len, :]

            # 计算weight并写入df
            if_treated_list = np.array(list(self.matched_data[self.control]))
            pro_list = np.array(list(self.matched_data["PROPENSITY"]))

            weights = if_treated_list / pro_list + (1 - if_treated_list) / (1 - pro_list)
            self.weighted_df["Weight"] = weights
            print("Done!")

        # SMPW-T
        elif method == "SMRW-T":
            print("\nWeighted processing ...", end=" ")
            matched_data_len = self.matched_data.shape[0]
            self.weighted_df = self.matched_data.iloc[0:matched_data_len, :]

            # 计算weight并写入df
            if_treated_list = np.array(list(self.matched_data[self.control]))
            pro_list = np.array(list(self.matched_data["PROPENSITY"]))

            weights = if_treated_list + (1 - if_treated_list) * pro_list / (1 - pro_list)
            self.weighted_df["Weight"] = weights
            print("Done!")

        # SMPW-T
        elif method == "SMRW-C":
            print("\nWeighted processing ...", end=" ")
            matched_data_len = self.matched_data.shape[0]
            self.weighted_df = self.matched_data.iloc[0:matched_data_len, :]

            # 计算weight并写入df
            if_treated_list = np.array(list(self.matched_data[self.control]))
            pro_list = np.array(list(self.matched_data["PROPENSITY"]))

            weights = (1 - if_treated_list) + if_treated_list * (1 - pro_list) / pro_list
            self.weighted_df["Weight"] = weights
            print("Done!")

        # OW
        elif method == "OW":
            print("\nWeighted processing ...", end=" ")
            matched_data_len = self.matched_data.shape[0]
            self.weighted_df = self.matched_data.iloc[0:matched_data_len, :]

            # 计算weight并写入df
            if_treated_list = np.array(list(self.matched_data[self.control]))
            pro_list = np.array(list(self.matched_data["PROPENSITY"]))

            weights = if_treated_list + (1 - if_treated_list) * pro_list / (1 - pro_list)
            self.weighted_df["Weight"] = weights
            print("Done!")

        elif method == "IPTW-P":
            print("\nWeighted processing ...", end=" ")
            matched_data_len = self.matched_data.shape[0]
            self.weighted_df = self.matched_data.iloc[0:matched_data_len, :]

            if_treated_list = np.array(list(self.matched_data[self.control]))
            pro_list = np.array(list(self.matched_data["PROPENSITY"]))

            for i in range(matched_data_len):
                if if_treated_list[i] == 0:
                    pro_list[i] = 1 - pro_list[i]
            pmin = [min(x, 1 - x) for x in pro_list]

            weights = pmin / pro_list
            self.weighted_df["Weight"] = weights
            print("Done!")

        # 若method非法，报错
        else:
            raise AttributeError(
                "\'method\' is invalid  when %s do weighted_process.\nHint:method -- \'IPTW\' \'SMRW-T\' \'SMRW-C\' \'OW\' \'IPTW-P\'" % (
                    self))


    def evaluate_p_value(self, df,if_show=True):
        '''
        Conducts chi-square tests to verify statistically that the cases/controls
        are well-matched on the variables of interest.
        Parameters
        ----------
        df : Pandas DataFrame
            df to evaluate p-value
        method: str
            Weight processing method, including 'IPTW' 'SMRW-T' 'SMRW-C' 'OW' 'IPTW-P'
        if_show: bool
            if show p-value condition
        Return
        ----------
        A dictionary contains the p-values of each variable and the number of variables that failed the test
        '''

        # Assert that the Match object has 'matches' and 'matched_data' attributes
        if not hasattr(self, 'df'):
            raise AttributeError("%s does not have a 'df' attribute." % (self))
        if not hasattr(self, 'covariants'):
            raise AttributeError("%s does not have a 'covariants' attribute." % (self))
        if not hasattr(self, 'control'):
            raise AttributeError("%s does not have a 'control' attribute." % (self))

        # Get variables of interest for analysis
        variables = self.covariants
        results = {}
        cnt = 0

        # 当df未经过weight处理时
        if "Weight" not in df:
            # Evaluate case/control match for each variable of interest
            from scipy.stats import chi2_contingency
            for var in variables:
                crosstable = pd.crosstab(df[self.control], df[var])

                # 如果var的唯一值<=2,进行2x2卡方检验
                if len(df[var].unique().tolist()) <= 2:
                    f_obs = np.array([crosstable.iloc[0][0:2].values,
                                      crosstable.iloc[1][0:2].values])
                    result = chi2_contingency(f_obs)[0:3]
                    round_result = (round(i, 4) for i in result)
                    p_val = list(round_result)[1]

                # 否则进行更大表格的卡方检验
                else:
                    C = crosstable.shape[1]
                    f_obs = np.array([crosstable.iloc[0][0:C].values,
                                      crosstable.iloc[1][0:C].values])
                    result = chi2_contingency(f_obs)[0:3]
                    round_result = (round(i, 4) for i in result)
                    p_val = list(round_result)[1]

                # 给results赋值并输出当前变量p值情况
                results[var] = p_val
                if if_show:
                    print("\t" + var + ": p_value = " + str(p_val), end="")
                if p_val < 0.01:
                    if if_show:
                        print(" FAILED")
                    cnt += 1
                else:
                    if if_show:
                        print(" PASSED")

        # 当df经过weight处理时
        else:
            df_control = df[df[self.control] == 0]
            df_treat = df[df[self.control] == 1]

            # Evaluate case/control match for each variable of interest
            from scipy.stats import chi2_contingency
            for var in variables:
                # 当变量为非连续变量时
                if isinstance(list(df[var])[0], str):

                    crosstable = pd.crosstab(df[self.control], df[var])
                    names = df[var].unique()
                    for name in names:
                        # 得到非连续值对应的weight列表
                        treat_name_weights = df_treat[df_treat[var] == name]["Weight"]
                        control_name_weights = df_control[df_control[var] == name]["Weight"]

                        # 计算频次
                        treat_name_times = np.sum(treat_name_weights)
                        control_name_times = np.sum(control_name_weights)

                        # 修改crosstab值
                        crosstable.loc[0, name] = treat_name_times
                        crosstable.loc[1, name] = control_name_times

                # 当变量为连续变量时
                else:
                    crosstable = pd.crosstab(df[self.control], df[var] * df['Weight'].copy())

                # 如果var的唯一值<=2,进行2x2卡方检验
                if len(df[var].unique().tolist()) <= 2:
                    f_obs = np.array([crosstable.iloc[0][0:2].values,
                                      crosstable.iloc[1][0:2].values])
                    result = chi2_contingency(f_obs)[0:3]
                    round_result = (round(i, 4) for i in result)
                    p_val = list(round_result)[1]

                # 否则进行更大表格的卡方检验
                else:
                    C = crosstable.shape[1]
                    f_obs = np.array([crosstable.iloc[0][0:C].values,
                                      crosstable.iloc[1][0:C].values])
                    result = chi2_contingency(f_obs)[0:3]
                    round_result = (round(i, 4) for i in result)
                    p_val = list(round_result)[1]

                # 输出当前变量的p值情况
                if if_show:
                    print("\t" + var + ": p_value = " + str(p_val), end="")
                if p_val < 0.01:
                    if if_show:
                        print(" FAILED")
                    cnt += 1
                else:
                    if if_show:
                        print(" PASSED")
                results[var] = p_val

        # 输出配对成功情况
        results["fail"] = cnt
        if cnt == 0:
            if if_show:
                print("\nAll variables were successfully matched!")
            return results
        else:
            if if_show:
                print("\n" + str(cnt) + " variable failed to match!")
            return results


    def evaluate_dependent(self, df):
        '''
        Evaluate the mean, variance, SMD of coveriates and the results of dependent variables
        Parameters
        ----------
        df : Pandas DataFrame
            The DataFrame to be evaluated
        Return
        ----------
        A Pandas DataFrame contains evaluate results
        '''

        # 判断变量是否齐全
        if not hasattr(self, 'df'):
            raise AttributeError("%s does not have a 'df' attribute." % (self))
        if not hasattr(self, 'covariants'):
            raise AttributeError("%s does not have a 'covariants' attribute." % (self))
        if not hasattr(self, 'control'):
            raise AttributeError("%s does not have a 'control' attribute." % (self))

        name_all = {}
        # 当df没经过weight处理时
        if "Weight" not in df:
            # 构建表格
            df_control = df[df[self.control] == 0]
            df_treat = df[df[self.control] == 1]
            indexl = ["Num"]
            for cov in self.covariants:
                indexl.append(cov)
                if isinstance(list(df[cov])[0], str):
                    n = []
                    name = df_treat[cov].value_counts(dropna=False, normalize=True)
                    for names in name.index:
                        if names in df_control[cov].values:
                            indexl.append("  " + str(names))
                            n.append(names)
                    name_all[cov] = n
            if self.dependents:
                for dep in self.dependents:
                    indexl.append(dep)
                    if isinstance(list(df[dep])[0], str):
                        n = []
                        name = df_treat[dep].value_counts(dropna=False, normalize=True)
                        for names in name.index:
                            if names in df_control[dep].values:
                                indexl.append("  " + str(names))
                                n.append(names)
                        name_all[dep] = n
            eva_df = pd.DataFrame(columns=["Treat", "Control", "smd"], index=indexl)

            # calculate and write number of treat and control objects
            treat_num = len(df_treat)
            control_num = len(df_control)
            eva_df.loc['Num', 'Treat'] = treat_num
            eva_df.loc['Num', 'Control'] = control_num

            # calculate and write mean and std of covariants
            for cov in self.covariants:
                # 当协变量为非连续变量时
                if isinstance(list(df[cov])[0], str):
                    ## 提取出变量的所有非连续值
                    name_treat = df_treat[cov].value_counts(dropna=False)
                    name_control = df_control[cov].value_counts(dropna=False)

                    for name in name_all[cov]:
                        eva_df.loc["  " + str(name), "Treat"] = str(name_treat[name]) + " (" + str(
                            round(name_treat[name] / treat_num, 3)) + ")"
                        eva_df.loc["  " + str(name), "Control"] = str(name_control[name]) + " (" + str(
                            round(name_control[name] / control_num, 3)) + ")"
                    p_treat = name_treat[0] / treat_num
                    p_control = name_control[0] / control_num
                    smd = 2 * (p_treat - p_control) / (p_treat * (1 - p_treat) + p_control * (1 - p_control))
                    eva_df.loc[cov, "smd"] = str(round(abs(smd), 3))

                # 当协变量为连续变量时
                else:
                    eva_df.loc[cov, "Treat"] = str(round(df_treat[cov].mean(), 3)) + " (" + str(
                        round(df_treat[cov].std(), 3)) + ")"
                    eva_df.loc[cov, "Control"] = str(round(df_control[cov].mean(), 3)) + " (" + str(
                        round(df_control[cov].std(), 3)) + ")"
                    smd = (df_treat[cov].mean() - df_control[cov].mean()) / math.sqrt(
                        (df_treat[cov].std() * df_treat[cov].std() + df_control[cov].std() * df_control[cov].std()) / 2)
                    eva_df.loc[cov, "smd"] = str(round(abs(smd), 3))
            # analysis dependents
            if self.dependents:
                for dep in self.dependents:
                    # 当分析变量为非连续变量时
                    if isinstance(list(df[dep])[0], str):
                        name_treat = df_treat[dep].value_counts(dropna=False)
                        name_control = df_control[dep].value_counts(dropna=False)

                        for name in name_all[dep]:
                            eva_df.loc["  " + name, "Treat"] = str(name_treat[name]) + " (" + str(
                                round(name_treat[name] / treat_num, 3)) + ")"
                            eva_df.loc["  " + name, "Control"] = str(name_control[name]) + " (" + str(
                                round(name_control[name] / control_num, 3)) + ")"

                    # 当分析变量为连续变量时
                    else:
                        eva_df.loc[dep, "Treat"] = str(round(df_treat[dep].mean(), 3)) + " (" + str(
                            round(df_treat[dep].std(), 3)) + ")"
                        eva_df.loc[dep, "Control"] = str(round(df_control[dep].mean(), 3)) + " (" + str(
                            round(df_control[dep].std(), 3)) + ")"

            # 完善表格结构并输出
            eva_df.replace({None: ""}, inplace=True)
            return eva_df

        # 当df经过weight处理时
        else:
            # 构建表格
            df_control = df[df[self.control] == 0]
            df_treat = df[df[self.control] == 1]
            indexl = ["Num"]
            for cov in self.covariants:
                indexl.append(cov)
                if isinstance(list(df[cov])[0], str):
                    n = []
                    name = df_treat[cov].value_counts(dropna=False, normalize=True)
                    for names in name.index:
                        if names in df_control[cov].values:
                            indexl.append("  " + str(names))
                            n.append(names)
                    name_all[cov] = n
            if self.dependents:
                for dep in self.dependents:
                    indexl.append(dep)
                    if isinstance(list(df[dep])[0], str):
                        n = []
                        name = df_treat[dep].value_counts(dropna=False, normalize=True)
                        for names in name.index:
                            if names in df_control[dep].values:
                                indexl.append("  " + str(names))
                                n.append(names)
                        name_all[dep] = n
            eva_df = pd.DataFrame(columns=["Treat", "Control", "smd"], index=indexl)

            # 提取出weight数据
            treat_weights = np.array(list(df_treat["Weight"]))
            control_weights = np.array(list(df_control["Weight"]))
            treat_num = len(df_treat)
            control_num = len(df_control)

            # calculate and write number of treat and control objects
            eva_df.loc['Num', 'Treat'] = treat_weights.sum()
            eva_df.loc['Num', 'Control'] = control_weights.sum()

            # calculate and write 频率和频次 of covariants
            for cov in self.covariants:
                # 当变量为非连续变量时
                if isinstance(list(df[cov])[0], str):

                    for name in name_all[cov]:
                        # 得到非连续值对应的weight列表
                        treat_name_weights = df_treat[df_treat[cov] == name]["Weight"]
                        control_name_weights = df_control[df_control[cov] == name]["Weight"]

                        # 计算频次频率
                        treat_name_times = np.sum(treat_name_weights)
                        control_name_times = np.sum(control_name_weights)
                        totol_times = treat_name_times + control_name_times
                        p_treat_name = treat_name_times / totol_times
                        p_control_name = control_name_times / totol_times

                        # 填入数据
                        eva_df.loc["  " + str(name), "Treat"] = str(round(treat_name_times, 3)) + " (" + str(
                            round(p_treat_name, 3)) + ")"
                        eva_df.loc["  " + str(name), "Control"] = str(round(control_name_times, 3)) + " (" + str(
                            round(p_control_name, 3)) + ")"

                    # 计算smd并填入数据
                    smd = 2 * (p_treat_name - p_control_name) / (
                            p_treat_name * (1 - p_treat_name) + p_control_name * (1 - p_control_name))
                    eva_df.loc[cov, "smd"] = str(round(abs(smd), 3))

                # 当变量为连续变量时
                else:
                    cov_treat = df_treat[cov]
                    cov_control = df_control[cov]

                    ##计算均值
                    treat_mean = np.average(cov_treat, weights=treat_weights)
                    control_mean = np.average(cov_control, weights=treat_weights)

                    ##计算方差
                    treat_var = np.average((cov_treat - treat_mean) ** 2, weights=treat_weights)
                    treat_std = np.sqrt(treat_var)
                    control_var = np.average((cov_control - control_mean) ** 2, weights=control_weights)
                    control_std = np.sqrt(control_var)

                    ##计算smd
                    smd = (treat_mean - control_mean) / np.sqrt((treat_std ** 2 + control_std ** 2) / 2)

                    ##填入数据
                    eva_df.loc[cov, "Treat"] = str(round(treat_mean, 3)) + " (" + str(round(treat_std, 3)) + ")"
                    eva_df.loc[cov, "Control"] = str(round(control_mean, 3)) + " (" + str(round(control_std, 3)) + ")"
                    eva_df.loc[cov, "smd"] = str(round(abs(smd), 3))

            # analysis dependents
            if self.dependents:
                for dep in self.dependents:
                    # 当变量为非连续变量时
                    if isinstance(list(df[dep])[0], str):
                        name_treat = df_treat[dep].value_counts(dropna=False)
                        name_control = df_control[dep].value_counts(dropna=False)

                        for name in name_all[dep]:
                            eva_df.loc["  " + name, "Treat"] = str(name_treat[name]) + " (" + str(
                                round(name_treat[name] / treat_num, 3)) + ")"
                            eva_df.loc["  " + name, "Control"] = str(name_control[name]) + " (" + str(
                                round(name_control[name] / control_num, 3)) + ")"

                    # 当变量为连续变量时
                    else:
                        dep_treat = df_treat[dep]
                        dep_control = df_control[dep]

                        ##计算均值
                        treat_mean = np.average(dep_treat, weights=treat_weights)
                        control_mean = np.average(dep_control, weights=treat_weights)

                        ##计算方差
                        treat_var = np.average((dep_treat - treat_mean) ** 2, weights=treat_weights)
                        treat_std = np.sqrt(treat_var)
                        control_var = np.average((dep_control - control_mean) ** 2, weights=control_weights)
                        control_std = np.sqrt(control_var)

                        ##计算smd
                        smd = (treat_mean - control_mean) / np.sqrt((treat_std ** 2 + control_std ** 2) / 2)

                        ##填入数据
                        eva_df.loc[dep, "Treat"] = str(round(treat_mean, 3)) + " (" + str(round(treat_std, 3)) + ")"
                        eva_df.loc[dep, "Control"] = str(round(control_mean, 3)) + " (" + str(
                            round(control_std, 3)) + ")"

            # 完善表格结构并输出
            eva_df.replace({None: ""}, inplace=True)
            return eva_df


    def plot_matching_efficiency(self,if_weighed=True):
        '''
        Draw a line chart of smd and a histogram of the number of variables that failed the test,
        showing the matching effect of each method
        Parameters
        ----------
        if_weighed: bool
            if weight processed
        '''

        # 检验参数是否齐全
        if not hasattr(self, 'df'):
            raise AttributeError("%s does not have a 'df' attribute." % (self))
        if not hasattr(self, 'matched_data'):
            raise AttributeError("%s does not have a 'matched_date' attribute." % (self))
        if if_weighed:
            if not hasattr(self, 'weighted_df'):
                raise AttributeError("%s does not have a 'weighted_df' attribute." % (self))
        v = []
        # plot smd
        plt.figure(figsize=(20, 8), dpi=300)
        df = self.evaluate_dependent(self.df)
        df = df[df['smd'] != '']['smd']
        values = [float(x) for x in df.values]
        plt.plot(df.index, values, linestyle='--', color='b', marker='o', label='Original Data')
        v.append(np.average(values))
        df = self.evaluate_dependent(self.matched_data)
        df = df[df['smd'] != '']['smd']
        values = [float(x) for x in df.values]
        plt.plot(df.index, values, linestyle='--', color='r', marker='o', label='Matched Data')
        v.append(np.average(values))
        if if_weighed:
            df = self.evaluate_dependent(self.weighted_df)
            df = df[df['smd'] != '']['smd']
            values = np.array([float(i) for i in df.values])
            plt.plot(df.index, values, linestyle='--', color='g', marker='o', label='Weighted Data')
            v.append(np.average(values))
        plt.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 8})
        plt.xticks(fontproperties='Times New Roman', fontsize=4, rotation=45)
        plt.yticks(fontproperties='Times New Roman', fontsize=4)
        titlename = "Smd for different method"
        plt.title(titlename, fontdict={'family': 'Times New Roman', 'size': 8})
        plt.xlabel('Covariants', fontdict={'family': 'Times New Roman', 'size': 8}, )
        plt.ylabel('smd', fontdict={'family': 'Times New Roman', 'size': 8})
        plt.show()
        print(v)

        plt.figure(figsize=(20, 8), dpi=300)
        index = ["Original Data", 'Matched Data']
        if if_weighed:
            index.append("Weighted Data")
        fail = []
        result = self.evaluate_p_value(self.df, if_show=False)
        fail.append(result["fail"])
        result = self.evaluate_p_value(self.matched_data, if_show=False)
        fail.append(result["fail"])
        if if_weighed:
            result = self.evaluate_p_value(self.weighted_df, if_show=False)
            fail.append(result["fail"])
        plt.bar(index, fail, facecolor='lightblue', edgecolor='black', alpha=0.8)
        plt.xticks(fontproperties='Times New Roman', fontsize=4)
        plt.yticks(fontproperties='Times New Roman', fontsize=4)
        titlename = "Failed p-value for different method"
        plt.title(titlename, fontdict={'family': 'Times New Roman', 'size': 8})
        plt.xlabel('Covariants', fontdict={'family': 'Times New Roman', 'size': 8}, )
        plt.ylabel('smd', fontdict={'family': 'Times New Roman', 'size': 8})
        print(fail)
        plt.show()


    def run(self):
        self.calculate_proprnsity_scores()
        self.match()
        self.evaluate_p_value()
        self.evaluate_dependent()