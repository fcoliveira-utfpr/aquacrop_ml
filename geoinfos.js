// ===============================
// LISTA DE MUNICÍPIOS
// ===============================
var municipios = [
  'Céu Azul',
  'Medianeira',
  'Missal',
  'Mercedes',
  'Sarandi',
  'Medianeira',
  'Maripá',
  'Palotina'
];

var nome_estado = 'Paraná';

// ===============================
// BASE GADM
// ===============================
var gadm = ee.FeatureCollection('projects/fcoliveira/assets/gadm41_BRA_2');

// ===============================
// SRTM
// ===============================
var srtm = ee.Image('USGS/SRTMGL1_003');

// ===============================
// FUNÇÃO PARA EXTRAIR DADOS
// ===============================
var extrairDados = function(nome_municipio) {
  
  var municipio = gadm.filter(ee.Filter.and(
    ee.Filter.eq('NAME_2', nome_municipio),
    ee.Filter.eq('NAME_1', nome_estado)
  )).first();
  
  // Verifica se encontrou o município
  var geom = ee.Feature(municipio).geometry();
  
  var centroid = geom.centroid();
  
  var lon = centroid.coordinates().get(0);
  var lat = centroid.coordinates().get(1);
  
  var elev = srtm.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: centroid,
    scale: 30
  }).get('elevation');
  
  return ee.Feature(null, {
    municipio: nome_municipio,
    estado: nome_estado,
    latitude: lat,
    longitude: lon,
    altitude_m: elev
  });
};

// ===============================
// APLICAR PARA TODOS
// ===============================
var listaFeatures = ee.FeatureCollection(
  ee.List(municipios).map(extrairDados)
);

// Visualizar
print(listaFeatures);

// ===============================
// EXPORTAR CSV
// ===============================
Export.table.toDrive({
  collection: listaFeatures,
  description: 'municipios_pr_altitude',
  fileFormat: 'CSV'
});
