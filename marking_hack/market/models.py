from django.db import models


class Region(models.Model):
    code = models.IntegerField(unique=True, primary_key=True, db_index=True)
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class City(models.Model):
    fias = models.IntegerField(db_index=True, primary_key=True)
    name = models.CharField(max_length=250)


class Store(models.Model):
    id_sp = models.CharField(max_length=32, unique=True, db_index=True)
    member = models.ForeignKey("Member", related_name="items", on_delete=models.CASCADE)
    region = models.ForeignKey(
        "Region",
        null=True,
        related_name="stores",
        on_delete=models.SET_NULL,
    )
    city = models.ForeignKey(
        "City",
        null=True,
        related_name="stores",
        on_delete=models.SET_NULL,
    )
    postal_code = models.IntegerField(null=True, blank=True)

    @property
    def region_code(self):
        return self.region_id


class Item(models.Model):
    gtin = models.CharField(max_length=32, unique=True, db_index=True)
    member = models.ForeignKey("Member", related_name="items", on_delete=models.CASCADE)
    product_name = models.CharField(max_length=250)
    product_short_name = models.CharField(max_length=250)
    tnved = models.IntegerField()
    tnved10 = models.IntegerField()
    brand = models.CharField(max_length=250)
    country = models.CharField(max_length=100)
    volume = models.IntegerField()

    def __str__(self):
        return self.product_name


class Member(models.Model):
    inn = models.IntegerField(unique=True, primary_key=True, db_index=True)
    region = models.ForeignKey(
        "Region",
        null=True,
        related_name="members",
        on_delete=models.SET_NULL,
    )


class StoreExport(models.Model):
    store = models.ForeignKey("Store", related_name="exports", on_delete=models.CASCADE)
    file = models.FileField(upload_to="exports/")

    def __str__(self):
        return f"export file from {self.store}"
