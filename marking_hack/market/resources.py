from import_export import fields, resources
from import_export.widgets import ForeignKeyWidget

from marking_hack.market.models import Store, Item, Region


class ItemResource(resources.ModelResource):
    store = fields.Field(
        column_name="store",
        attribute="store",
        widget=ForeignKeyWidget(Store, field="id"),
    )

    class Meta:
        model = Item
        fields = ("store", "name", "amount")


class StoreResource(resources.ModelResource):
    region = fields.Field(
        column_name="region",
        attribute="region",
        widget=ForeignKeyWidget(Region, field="code"),
    )

    class Meta:
        model = Item
        fields = ("name", "region", "postal_code")
