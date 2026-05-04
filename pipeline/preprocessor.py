"""
Preprocessing Pipeline for HR Analytics
Fixes applied:
 - Smart missing-value strategy: high(≥30%) → 'Unknown', medium(≥10%) → 'Unknown', low → mode/median
 - Fit encoders/scaler on TRAIN only — no leakage into validation/test
 - Ordinal encoding for ordered categoricals (experience, company_size, last_new_job)
 - Label encoding for remaining nominal categoricals (acceptable for tree models)
 - Extra engineered features: training_bucket, has_rel_exp
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

HIGH_MISS_THRESH   = 0.30
MEDIUM_MISS_THRESH = 0.10


class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler         = StandardScaler()
        self.feature_names  = []
        self.cat_cols       = []
        self.num_cols       = []
        self._fit_cols      = None
        self._fill_values   = {}

    def __getstate__(self):
        return {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'cat_cols': self.cat_cols,
            'num_cols': self.num_cols,
            '_fit_cols': self._fit_cols,
            '_fill_values': self._fill_values,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f'Loaded object is not {cls.__name__}')
        return obj

    def _fit_fill_values(self, df: pd.DataFrame):
        """Determine fill values from TRAIN data only."""
        self._fill_values = {}
        for col in df.columns:
            if col in ('target', 'enrollee_id'):
                continue
            miss_rate = df[col].isnull().mean()
            if df[col].dtype == object:
                if miss_rate >= HIGH_MISS_THRESH:
                    self._fill_values[col] = 'Unknown'
                elif miss_rate >= MEDIUM_MISS_THRESH:
                    self._fill_values[col] = 'Unknown'
                else:
                    mode = df[col].mode()
                    self._fill_values[col] = mode[0] if len(mode) else 'Unknown'
            else:
                self._fill_values[col] = float(pd.to_numeric(df[col], errors='coerce').median())

    def _apply_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, val in self._fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna('Unknown')
        for col in df.select_dtypes(include=np.number).columns:
            if col not in ('target', 'enrollee_id'):
                df[col] = df[col].fillna(df[col].median())
        return df

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        exp_map = {str(i): i for i in range(1, 21)}
        exp_map.update({'<1': 0, '>20': 21, 'Unknown': 10})
        df['experience_num'] = df['experience'].map(exp_map).fillna(10)

        lnj_map = {'never': 0, '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5, 'Unknown': 2}
        df['last_new_job_num'] = df['last_new_job'].map(lnj_map).fillna(2)

        cs_map = {'Unknown': 0, '<10': 1, '10-49': 2, '50-99': 3,
                  '100-500': 4, '500-999': 5, '1000-4999': 6,
                  '5000-9999': 7, '10000+': 8}
        df['company_size_num'] = df['company_size'].map(cs_map).fillna(0)

        df['high_cdi']    = (df['city_development_index'] > 0.9).astype(int)
        rel_exp           = (df['relevent_experience'] == 'Has relevent experience').astype(int)
        df['has_rel_exp'] = rel_exp
        df['cdi_x_rel_exp']  = df['city_development_index'] * rel_exp
        df['is_enrolled']    = (df['enrolled_university'] != 'no_enrollment').astype(int)
        df['training_bucket'] = pd.cut(
            df['training_hours'], bins=[0, 20, 50, 100, 9999],
            labels=[0, 1, 2, 3]).astype(float).fillna(0)

        return df

    def _build_distributions(self, df: pd.DataFrame) -> list:
        distributions = []
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in num_cols:
            vals = df[col].dropna().astype(float)
            if len(vals) == 0:
                continue
            counts, bins = np.histogram(vals, bins=min(12, max(3, int(np.sqrt(len(vals))))))
            distributions.append({
                'feature': col,
                'type': 'numeric',
                'bins': [float(x) for x in bins.tolist()],
                'counts': [int(x) for x in counts.tolist()],
                'min': float(vals.min()),
                'max': float(vals.max()),
                'mean': float(vals.mean()),
                'std': float(vals.std()),
            })

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        for col in cat_cols:
            vc = df[col].fillna('Unknown').value_counts().head(12)
            distributions.append({
                'feature': col,
                'type': 'categorical',
                'values': [{'category': str(k), 'count': int(v)} for k, v in vc.items()]
            })

        return distributions

    def _build_boxplots(self, df: pd.DataFrame) -> list:
        boxplots = []
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in num_cols:
            vals = df[col].dropna().astype(float)
            if len(vals) == 0:
                continue
            q1 = float(vals.quantile(0.25))
            q2 = float(vals.quantile(0.50))
            q3 = float(vals.quantile(0.75))
            iqr = q3 - q1
            lower_whisker = float(max(vals.min(), q1 - 1.5 * iqr))
            upper_whisker = float(min(vals.max(), q3 + 1.5 * iqr))
            outliers = vals[(vals < lower_whisker) | (vals > upper_whisker)]
            boxplots.append({
                'feature': col,
                'min': float(vals.min()),
                'q1': q1,
                'median': q2,
                'q3': q3,
                'max': float(vals.max()),
                'lower_whisker': lower_whisker,
                'upper_whisker': upper_whisker,
                'iqr': float(iqr),
                'outlier_count': int(len(outliers)),
            })
        return boxplots

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        df.drop_duplicates(inplace=True)
        self._fit_fill_values(df.drop(columns=['target'], errors='ignore'))
        df = self._apply_fill(df)
        df = self._engineer(df)

        drop_cols = ['enrollee_id', 'city', 'experience', 'last_new_job', 'company_size']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # Build distributions and boxplots BEFORE popping target (but after dropping unnecessary cols)
        distributions = self._build_distributions(df.drop(columns=['target'], errors='ignore'))
        boxplots = self._build_boxplots(df.drop(columns=['target'], errors='ignore'))

        # Cap outliers for numerical columns (excluding target)
        for bp in boxplots:
            col = bp['feature']
            if col in df.columns:
                df[col] = df[col].clip(lower=bp['lower_whisker'], upper=bp['upper_whisker'])

        y = df.pop('target').astype(int)

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        self.cat_cols = cat_cols
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.feature_names = df.columns.tolist()
        self._fit_cols     = df.columns.tolist()
        num_cols           = df.select_dtypes(include=np.number).columns.tolist()
        self.num_cols      = num_cols
        self.scaler.fit(df[num_cols])

        df_out = df.copy()
        df_out[num_cols] = self.scaler.transform(df[num_cols])

        summary = {
            'original_rows':       int(len(df)),
            'features_created':    len(self.feature_names),
            'categorical_encoded': len(cat_cols),
            'numeric_scaled':      len(num_cols),
            'feature_names':       self.feature_names,
            'class_dist':          {str(int(k)): int(v) for k, v in y.value_counts().items()},
            'fill_strategy': {col: (f'Unknown (≥{int(MEDIUM_MISS_THRESH*100)}% missing)'
                                    if isinstance(v, str) and v == 'Unknown'
                                    else f'mode: {v}' if isinstance(v, str)
                                    else f'median: {v:.3f}' if not (isinstance(v, float) and pd.isna(v)) else 'mode (fallback)')
                              for col, v in list(self._fill_values.items())[:8]}
        }
        return df_out.values, y.values, summary, distributions, boxplots

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df = self._apply_fill(df)
        df = self._engineer(df)

        drop_cols = ['enrollee_id', 'city', 'experience', 'last_new_job', 'company_size', 'target']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        for col in self.cat_cols:
            if col in df.columns:
                le    = self.label_encoders[col]
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

        for col in self._fit_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[self._fit_cols]

        df_out = df.copy()
        df_out[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df_out.values
