
import pandas as pd
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2

    def fit(self, X_1, X_2, y):
        # train model_1
        self.model_1.fit(X_1, y)

        # make predictions
        y_fit = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=y.columns,
        )

        # compute residuals
        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze() # wide to long

        # train model_2 on residuals
        self.model_2.fit(X_2, y_resid)

        # save column names for predict method
        self.y_columns = y.columns
        # Save data for question checking
        self.y_fit = y_fit
        self.y_resid = y_resid
    def predict(self, X_1, X_2):
        # Predict with model_1
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns,
        )
        y_pred = y_pred.stack().squeeze()  # wide to long

        # Add model_2 predictions to model_1 predictions
        y_pred += self.model_2.predict(X_2)

        return y_pred.unstack()


class HierarchyModel():
    def __init__(self, geo_model, hybrid_model):
        self.geo_model = geo_model
        self.hybrid_model = hybrid_model

    def fit(self, X_1, X_2,X_geo, y_over_all, y_geo ):
        self.hybrid_model.fit(X_1, X_2, y_over_all)
        y_over_all_preds_fit= self.hybrid_model.predict(X_1, X_2)
        self.y_res_for_geo_train =self.get_geo_res( y_geo, y_over_all_preds_fit)
        self.geo_model.fit(X_geo, self.y_res_for_geo_train)

    def get_geo_res(self, y_geo, y_over_all):
        y_over_all_unstuck = y_over_all.unstack().rename('sales_overall').to_frame()
        y_res_for_geo = (y_geo
        .reset_index('store_id')
        .merge(y_over_all_unstuck, on=['sku_category', 'date'])
        .set_index('store_id', append=True)
        .assign(sale_res_geo=lambda df: df['sales_overall'] - df['sales'])
        .reset_index()
        .set_index(['date'])
        ['sale_res_geo']
        )
        return y_res_for_geo
    def predict(self, X_1, X_2,X_geo):
        y_over_all_pred = self.hybrid_model.predict(X_1, X_2)
        res_geo_preds = self.geo_model.predict(X_geo)
        y_geo_pred= self.get_per_geo_pred(X_geo, y_over_all_pred,res_geo_preds)
        return y_geo_pred

    def get_per_geo_pred(self, X_geo, y_over_all_pred, res_geo_preds):
        y_over_all_pred =y_over_all_pred.unstack().rename('sales_overall_pred').to_frame()
        pred_sales_per_geo = (
            X_geo
        .set_index(['store_id', 'sku_category'], append=True)
        .assign(preds_res=res_geo_preds)[['preds_res']]
        .reset_index()
        .merge(y_over_all_pred, on=['sku_category', 'date'])
        .assign(sales_per_geo=lambda df: df['sales_overall_pred'] - df['preds_res'])
        .set_index(['store_id', 'sku_category','date',])
        ['sales_per_geo']
        )
        return pred_sales_per_geo

def plot_val_prediction_and_train(y_sku_sales,y_fit,y_pred):
    sku_categories = y_sku_sales.columns[:10]
    axs = y_sku_sales.loc(axis=1)[sku_categories].plot(subplots=True,
                                                       sharex=True,
                                                       figsize=(40, 30),
                                                       color="0.75",
                                                       style=".-",
                                                       markeredgecolor="0.25",
                                                       markerfacecolor="0.25",
                                                       alpha=0.5)
    _ = y_fit.loc(axis=1)[sku_categories].plot(subplots=True, sharex=True, color='C0', ax=axs)
    _ = y_pred.loc(axis=1)[sku_categories].plot(subplots=True, sharex=True, color='C3', ax=axs)
    for ax, sku_categories in zip(axs, sku_categories):
        ax.legend([])
        ax.set_ylabel(sku_categories)

