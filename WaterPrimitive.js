const WaterMaterialSource = `
uniform sampler2D texture;

uniform sampler2D normalTexture;
uniform float time;

uniform mat4 fixedFrameToEastNorthUpTransform;

varying vec4 v_worldPosition;
varying vec4 v_uv;

uniform float size;
uniform vec4 waterColor;
uniform float waterAlpha;
uniform float rf0;
uniform vec3 lightDirection;
uniform float sunShiny;
uniform float distortionScale;

vec3 sunDirection = normalize( lightDirection );
vec3 sunColor = vec3( 1.0 );


vec4 getNoise( sampler2D normalMap, vec2 uv ) {
    vec2 uv0 = ( uv / 103.0 ) + vec2( time / 17.0, time / 29.0 );
    vec2 uv1 = uv / 107.0 - vec2( time / -19.0, time / 31.0 );
    vec2 uv2 = uv / vec2( 8907.0, 9803.0 ) + vec2( time / 101.0, time / 97.0 );
    vec2 uv3 = uv / vec2( 1091.0, 1027.0 ) - vec2( time / 109.0, time / -113.0 );
    vec4 noise = texture2D( normalMap, uv0 ) +
        texture2D( normalMap, uv1 ) +
        texture2D( normalMap, uv2 ) +
        texture2D( normalMap, uv3 );
    return noise * 0.5 - 1.0;
}

void sunLight( const vec3 surfaceNormal, const vec3 eyeDirection, float shiny, float spec, float diffuse, inout vec3 diffuseColor, inout vec3 specularColor ) {
    vec3 reflection = normalize( reflect( -sunDirection, surfaceNormal ) );
    float direction = max( 0.0, dot( eyeDirection, reflection ) );
    specularColor += pow( direction, shiny ) * sunColor * spec;
    diffuseColor += max( dot( sunDirection, surfaceNormal ), 0.0 ) * sunColor * diffuse;
}

czm_material czm_getMaterial(czm_materialInput materialInput) {
    czm_material material = czm_getDefaultMaterial(materialInput);

    vec2 transformedSt = materialInput.st * 2.0 - 1.0;
    vec4 noise = getNoise( normalTexture, transformedSt * size );
    vec3 surfaceNormal = normalize( noise.xzy );

    vec3 diffuseLight = vec3( 0.0 );
    vec3 specularLight = vec3( 0.0 );

    vec3 eye = ( czm_inverseView * vec4( vec3(0.0), 1.0 ) ).xyz;
    eye = ( fixedFrameToEastNorthUpTransform * vec4( eye, 1.0) ).xyz;
    vec3 world = ( fixedFrameToEastNorthUpTransform * vec4( v_worldPosition.xyz, 1.0) ).xyz;

    vec3 worldToEye = eye - world;
    worldToEye = vec3( worldToEye.x, worldToEye.z, -worldToEye.y );
    vec3 eyeDirection = normalize( worldToEye );

    float shiny = sunShiny;
    float spec = 2.0;
    float diffuse = 0.5;
    sunLight( surfaceNormal, eyeDirection, shiny, spec, diffuse, diffuseLight, specularLight );

    float distance = length( worldToEye );
    float distortionScale = distortionScale;
    vec2 distortion = surfaceNormal.xz * ( 0.001 + 1.0 / distance ) * distortionScale;
    vec3 reflectionSample = vec3( texture2D( texture, (v_uv.xy / v_uv.w) * 0.5 + 0.5 + distortion ) );

    float theta = max( dot( eyeDirection, surfaceNormal ), 0.0 );
    float rf0 = rf0;
    float reflectance = mix( rf0, 1.0, pow( 1.0 - theta, 5.0 ) );

    vec3 waterColor = waterColor.rgb;

    vec3 scatter = max( 0.0, dot( surfaceNormal, eyeDirection ) ) * waterColor;
    vec3 albedo = mix(
        sunColor * diffuseLight * 0.3 + scatter,
        vec3( 0.1 ) + reflectionSample * 0.9 + reflectionSample * specularLight,
        reflectance
    );
    material.diffuse = albedo.rgb;
    material.alpha = waterAlpha;

    return material;
}
`;
const WaterAppearanceVS = `
attribute vec3 position3DHigh;
attribute vec3 position3DLow;
attribute vec3 normal;
attribute vec2 st;
attribute float batchId;

varying vec3 v_positionEC;
varying vec3 v_normalEC;
varying vec2 v_st;

uniform mat4 reflectorProjectionMatrix;
uniform mat4 reflectorViewMatrix;
uniform mat4 reflectMatrix;
varying vec4 v_worldPosition;
varying vec4 v_uv;


void main()
{
    vec4 p = czm_computePosition();

    v_positionEC = (czm_modelViewRelativeToEye * p).xyz;
    v_normalEC = czm_normal * normal;
    v_st = st;

    mat4 modelView = reflectorViewMatrix * reflectMatrix * czm_model;
    modelView[3][0] = 0.0;
    modelView[3][1] = 0.0;
    modelView[3][2] = 0.0;
    v_uv = reflectorProjectionMatrix * modelView * p;
    vec4 positionMC = vec4( position3DHigh + position3DLow, 1.0 );
    v_worldPosition = czm_model * positionMC;

    gl_Position = czm_modelViewProjectionRelativeToEye * p;
}
`;

function createPlaceHolderTexture(context) {
  const placeholderTexture = new Cesium.Texture({
    context,
    source: {
      width: 1,
      height: 1,
      arrayBufferView: new Uint8Array([255, 0, 0, 255]),
    },
    sampler: new Cesium.Sampler({
      wrapS: Cesium.TextureWrap.REPEAT,
      wrapT: Cesium.TextureWrap.REPEAT,
      minificationFilter: Cesium.TextureMinificationFilter.LINEAR,
      magnificationFilter: Cesium.TextureMinificationFilter.LINEAR,
    }),
  });
  placeholderTexture.type = "sampler2D";

  return placeholderTexture;
}

function reflect(view, normal) {
  const scaledNormal = normal.clone();
  const reflect = view.clone();

  const scalar = 2 * Cesium.Cartesian3.dot(view, normal);
  Cesium.Cartesian3.multiplyByScalar(normal, scalar, scaledNormal);

  return Cesium.Cartesian3.subtract(view, scaledNormal, reflect);
}

function isPowerOfTwo(value) {
  return (value & (value - 1)) === 0 && value !== 0;
}

function addTextureUniform(options) {
  const { context, material, uniformName, imgSrc } = options;
  const wrapS = options.wrapS || Cesium.TextureWrap.REPEAT;
  const wrapT = options.wrapT || Cesium.TextureWrap.REPEAT;
  const minificationFilter =
    options.minificationFilter || Cesium.TextureMinificationFilter.LINEAR;
  const magnificationFilter =
    options.magnificationFilter || Cesium.TextureMagnificationFilter.LINEAR;

  const img = new Image();
  img.src = imgSrc;
  img.addEventListener("load", () => {
    const texture = new Cesium.Texture({
      context,
      source: img,
      sampler: new Cesium.Sampler({
        wrapS,
        wrapT,
        minificationFilter,
        magnificationFilter,
      }),
    });
    texture.type = "sampler2D";
    if (isPowerOfTwo(img.width) && isPowerOfTwo(img.height)) {
      texture.generateMipmap(Cesium.MipmapHint.NICEST);
    }
    material.uniforms[uniformName] = texture;
  });
}

const renderTilesetPassState = new Cesium.Cesium3DTilePassState({
  pass: Cesium.Cesium3DTilePass.RENDER,
});

const scratchBackgroundColor = new Cesium.Color();

function render(scene, passStateFramebuffer) {
  const frameState = scene._frameState;
  const context = scene.context;
  const us = context.uniformState;

  const view = scene._defaultView;
  scene._view = view;

  scene.updateFrameState();
  frameState.passes.render = true;
  frameState.passes.postProcess = scene.postProcessStages.hasSelected;
  frameState.tilesetPassState = renderTilesetPassState;

  let backgroundColor = Cesium.defaultValue(scene.backgroundColor, Cesium.Color.BLACK);
  if (scene._hdr) {
    backgroundColor = Cesium.Color.clone(backgroundColor, scratchBackgroundColor);
    backgroundColor.red = Math.pow(backgroundColor.red, scene.gamma);
    backgroundColor.green = Math.pow(backgroundColor.green, scene.gamma);
    backgroundColor.blue = Math.pow(backgroundColor.blue, scene.gamma);
  }
  frameState.backgroundColor = backgroundColor;

  scene.fog.update(frameState);

  us.update(frameState);

  const shadowMap = scene.shadowMap;
  if (Cesium.defined(shadowMap) && shadowMap.enabled) {
    if (!Cesium.defined(scene.light) || scene.light instanceof Cesium.SunLight) {
      Cesium.Cartesian3.negate(us.sunDirectionWC, scene._shadowMapCamera.direction);
    } else {
      Cesium.Cartesian3.clone(scene.light.direction, scene._shadowMapCamera.direction);
    }
    frameState.shadowMaps.push(shadowMap);
  }

  scene._computeCommandList.length = 0;
  scene._overlayCommandList.length = 0;

  const viewport = view.viewport;
  viewport.x = 0;
  viewport.y = 0;
  viewport.width = context.drawingBufferWidth;
  viewport.height = context.drawingBufferHeight;

  const passState = view.passState;
  passState.framebuffer = passStateFramebuffer;
  passState.blendingEnabled = undefined;
  passState.scissorTest = undefined;
  passState.viewport = Cesium.BoundingRectangle.clone(viewport, passState.viewport);

  if (Cesium.defined(scene.globe)) {
    scene.globe.beginFrame(frameState);
  }

  scene.updateEnvironment();
  scene.updateAndExecuteCommands(passState, backgroundColor);
  scene.resolveFramebuffers(passState);

  if (Cesium.defined(scene.globe)) {
    scene.globe.endFrame(frameState);

    if (!scene.globe.tilesLoaded) {
      scene._renderRequested = true;
    }
  }

  context.endFrame();
}

const clipBias = 0;

export default class WaterPrimitive {

  constructor(options) {
    this._scene = options.scene;
    this._height = options.height;
    this._flowDegrees = Cesium.defaultValue(options.flowDegrees, 0);

    const positions = options.positions;
    const total = positions.length;
    let x = 0;
    let y = 0;
    let z = 0;
    this._positions = [];
    positions.forEach((p) => {
      const lat = p.latitude;
      const lon = p.longitude;
      x += Math.cos(lat) * Math.cos(lon)
      y += Math.cos(lat) * Math.sin(lon)
      z += Math.sin(lat)
      this._positions.push(
        Cesium.Cartesian3.fromRadians(p.longitude, p.latitude, this._height)
      );
    });

    x /= total;
    y /= total;
    z /= total;
    const centerLon = Math.atan2(y, x)
    const hyp = Math.sqrt(x * x + y * y)
    const centerLat = Math.atan2(z, hyp)

    this._reflectorWorldPosition = Cesium.Cartesian3.fromRadians(centerLon, centerLat, this._height);
    this._originalreflectorWorldPosition = this._reflectorWorldPosition.clone();
    this._normal = Cesium.Ellipsoid.WGS84.geodeticSurfaceNormal(this._reflectorWorldPosition);
    this._waterPlane = Cesium.Plane.fromPointNormal(
      this._reflectorWorldPosition,
      this._normal
    );

    this._reflectMatrix = new Cesium.Matrix4(
      -2 * this._waterPlane.normal.x * this._waterPlane.normal.x + 1,
      -2 * this._waterPlane.normal.x * this._waterPlane.normal.y,
      -2 * this._waterPlane.normal.x * this._waterPlane.normal.z,
      -2 * this._waterPlane.normal.x * this._waterPlane.distance,
      -2 * this._waterPlane.normal.y * this._waterPlane.normal.x,
      -2 * this._waterPlane.normal.y * this._waterPlane.normal.y + 1,
      -2 * this._waterPlane.normal.y * this._waterPlane.normal.z,
      -2 * this._waterPlane.normal.y * this._waterPlane.distance,
      -2 * this._waterPlane.normal.z * this._waterPlane.normal.x,
      -2 * this._waterPlane.normal.z * this._waterPlane.normal.y,
      -2 * this._waterPlane.normal.z * this._waterPlane.normal.z + 1,
      -2 * this._waterPlane.normal.z * this._waterPlane.distance,
      0,
      0,
      0,
      1
    );

    this._reflectorViewMatrix = Cesium.Matrix4.IDENTITY.clone();
    this._reflectorProjectionMatrix = Cesium.Matrix4.IDENTITY.clone();

    this._initUniforms = {
      normalMapUrl: Cesium.defaultValue(
        options.normalMapUrl,
        "https://threejs.org/examples/textures/waternormals.jpg"
      ),
      size: Cesium.defaultValue(options.rippleSize, 50.0),
      waterColor: Cesium.defaultValue(
        options.waterColor,
        Cesium.Color.fromCssColorString("#001e0f")
      ),
      waterAlpha: Cesium.defaultValue(options.waterAlpha, 0.9),
      rf0: Cesium.defaultValue(options.reflectivity, 0.3),
      lightDirection: Cesium.defaultValue(
        options.lightDirection,
        new Cesium.Cartesian3(0, 0, 1)
      ),
      sunShiny: Cesium.defaultValue(options.sunShiny, 100.0),
      distortionScale: Cesium.defaultValue(options.distortionScale, 3.7),
    };

    const context = this._scene.context;
    this._createFramebuffer(
      context,
      context.drawingBufferWidth,
      context.drawingBufferHeight,
      this._scene.highDynamicRange
    );

    this._primitive = this._createPrimitive(
      this._positions,
      this._height,
      this._flowDegrees,
    );

    this._scene.primitives.add(this._primitive);

    this.preRender = this.preRender.bind(this);
    this._scene.preRender.addEventListener(this.preRender);

    Cesium.UniformState.prototype.updateFrustum = function (frustum) {
      Cesium.Matrix4.clone(
        Cesium.defaultValue(
          frustum.customProjectionMatrix,
          frustum.projectionMatrix
        ),
        this._projection
      );
      this._inverseProjectionDirty = true;
      this._viewProjectionDirty = true;
      this._inverseViewProjectionDirty = true;
      this._modelViewProjectionDirty = true;
      this._modelViewProjectionRelativeToEyeDirty = true;
      if (Cesium.defined(frustum.infiniteProjectionMatrix)) {
        Cesium.Matrix4.clone(
          frustum.infiniteProjectionMatrix,
          this._infiniteProjection
        );
        this._modelViewInfiniteProjectionDirty = true;
      }
      this._currentFrustum.x = frustum.near;
      this._currentFrustum.y = frustum.far;

      this._farDepthFromNearPlusOne = frustum.far - frustum.near + 1.0;
      this._log2FarDepthFromNearPlusOne = Cesium.Math.log2(
        this._farDepthFromNearPlusOne
      );
      this._oneOverLog2FarDepthFromNearPlusOne =
        1.0 / this._log2FarDepthFromNearPlusOne;

      if (Cesium.defined(frustum._offCenterFrustum)) {
        frustum = frustum._offCenterFrustum;
      }

      this._frustumPlanes.x = frustum.top;
      this._frustumPlanes.y = frustum.bottom;
      this._frustumPlanes.z = frustum.left;
      this._frustumPlanes.w = frustum.right;
    };

    Cesium.PerspectiveFrustum.prototype.clone = function (result) {
      if (!Cesium.defined(result)) {
        result = new Cesium.PerspectiveFrustum();
      }

      result.aspectRatio = this.aspectRatio;
      result.fov = this.fov;
      result.near = this.near;
      result.far = this.far;

      result._aspectRatio = undefined;
      result._fov = undefined;
      result._near = undefined;
      result._far = undefined;

      this._offCenterFrustum.clone(result._offCenterFrustum);

      result.customProjectionMatrix = this.customProjectionMatrix;

      return result;
    };
  }

  get rippleSize() {
    return this._material.uniforms.size;
  }
  set rippleSize(value) {
    this._material.uniforms.size = value;
  }

  get waterAlpha() {
    return this._material.uniforms.waterAlpha;
  }
  set waterAlpha(value) {
    this._material.uniforms.waterAlpha = value;
  }

  get reflectivity() {
    return this._material.uniforms.rf0;
  }
  set reflectivity(value) {
    this._material.uniforms.rf0 = value;
  }

  get distortionScale() {
    return this._material.uniforms.distortionScale;
  }
  set distortionScale(value) {
    this._material.uniforms.distortionScale = value;
  }

  get height() {
    return this._height;
  }
  set height(value) {
    this._height = value;

    const rwpCa = Cesium.Cartographic.fromCartesian(this._originalreflectorWorldPosition);
    const newRwpCa = Cesium.Cartesian3.fromRadians(rwpCa.longitude, rwpCa.latitude, this._height);
    const move = Cesium.Cartesian3.subtract(newRwpCa, this._originalreflectorWorldPosition, new Cesium.Cartesian3());
    const moveMatrix4 = Cesium.Matrix4.fromTranslation(move);
    this._primitive.modelMatrix = moveMatrix4;

    this._reflectorWorldPosition = newRwpCa;
    this._normal = Cesium.Ellipsoid.WGS84.geodeticSurfaceNormal(this._reflectorWorldPosition);
    this._waterPlane = Cesium.Plane.fromPointNormal(
      this._reflectorWorldPosition,
      this._normal
    );

    this._reflectMatrix = new Cesium.Matrix4(
      -2 * this._waterPlane.normal.x * this._waterPlane.normal.x + 1,
      -2 * this._waterPlane.normal.x * this._waterPlane.normal.y,
      -2 * this._waterPlane.normal.x * this._waterPlane.normal.z,
      -2 * this._waterPlane.normal.x * this._waterPlane.distance,
      -2 * this._waterPlane.normal.y * this._waterPlane.normal.x,
      -2 * this._waterPlane.normal.y * this._waterPlane.normal.y + 1,
      -2 * this._waterPlane.normal.y * this._waterPlane.normal.z,
      -2 * this._waterPlane.normal.y * this._waterPlane.distance,
      -2 * this._waterPlane.normal.z * this._waterPlane.normal.x,
      -2 * this._waterPlane.normal.z * this._waterPlane.normal.y,
      -2 * this._waterPlane.normal.z * this._waterPlane.normal.z + 1,
      -2 * this._waterPlane.normal.z * this._waterPlane.distance,
      0,
      0,
      0,
      1
    );
  }

  _createReflectionWaterMaterial() {
    const context = this._scene.context;

    const placeholderTexture = createPlaceHolderTexture(context);
    const {
      normalMapUrl,
      size,
      waterColor,
      waterAlpha,
      rf0,
      lightDirection,
      sunShiny,
      distortionScale,
    } = this._initUniforms;

    const texture = Cesium.Texture.fromFramebuffer({
      context,
      framebuffer: this._colorFramebuffer,
    });
    texture.type = "sampler2D";

    const initUniforms = {
      size: size,
      waterColor: waterColor,
      waterAlpha: waterAlpha,
      rf0: rf0,
      lightDirection: lightDirection,
      sunShiny: sunShiny,
      distortionScale: distortionScale,
      normalTexture: placeholderTexture,
      texture: texture,
      time: 0,
      fixedFrameToEastNorthUpTransform: Cesium.Matrix4.toArray(
        this._getFixedFrameToEastNorthUpTransformFromWorldMatrix()
      ),
    };
    const material = new Cesium.Material({
      fabric: {
        type: "ReflectionWater",
        uniforms: initUniforms,
        source: WaterMaterialSource,
      },
      translucent: false,
      minificationFilter: Cesium.TextureMinificationFilter.LINEAR,
      magnificationFilter: Cesium.TextureMagnificationFilter.LINEAR,
    });

    addTextureUniform({
      context,
      material,
      uniformName: "normalTexture",
      imgSrc: normalMapUrl,
    });
    return material;
  }

  _updateVirtualCamera(camera) {
    let lookAtPosition = new Cesium.Cartesian3(0, 0, -1);
    let target = new Cesium.Cartesian3();

    this._virtualCamera = Cesium.Camera.clone(
      camera,
      this._virtualCamera
    );

    const cameraWorldPosition = camera.positionWC.clone();

    let view = Cesium.Cartesian3.subtract(
      this._reflectorWorldPosition,
      cameraWorldPosition,
      new Cesium.Cartesian3()
    );

    if (Cesium.Cartesian3.dot(view, this._normal) > 0) {
      return false;
    }

    view = reflect(view, this._normal);
    Cesium.Cartesian3.negate(view, view);

    Cesium.Cartesian3.add(view, this._reflectorWorldPosition, view);
    this._virtualCamera.position = view.clone();

    Cesium.Cartesian3.add(
      camera.directionWC,
      cameraWorldPosition,
      lookAtPosition
    );

    Cesium.Cartesian3.subtract(
      this._reflectorWorldPosition,
      lookAtPosition,
      target
    );
    target = reflect(target, this._normal);
    Cesium.Cartesian3.negate(target, target);
    Cesium.Cartesian3.add(target, this._reflectorWorldPosition, target);

    this._virtualCamera.direction = Cesium.Cartesian3.subtract(
      target,
      this._virtualCamera.position,
      new Cesium.Cartesian3()
    );
    Cesium.Cartesian3.normalize(
      this._virtualCamera.direction,
      this._virtualCamera.direction
    );

    Cesium.Cartesian3.add(camera.upWC, cameraWorldPosition, lookAtPosition);

    Cesium.Cartesian3.subtract(
      this._reflectorWorldPosition,
      lookAtPosition,
      target
    );
    target = reflect(target, this._normal);
    Cesium.Cartesian3.negate(target, target);
    Cesium.Cartesian3.add(target, this._reflectorWorldPosition, target);

    this._virtualCamera.up = Cesium.Cartesian3.subtract(
      target,
      this._virtualCamera.position,
      new Cesium.Cartesian3()
    );
    Cesium.Cartesian3.normalize(this._virtualCamera.up, this._virtualCamera.up);

    this._reflectorProjectionMatrix = this._virtualCamera.frustum.projectionMatrix;
    this._reflectorViewMatrix = this._virtualCamera.viewMatrix;

    const reflectorPlane = Cesium.Plane.fromPointNormal(
      this._reflectorWorldPosition,
      this._normal
    );
    Cesium.Plane.transform(
      reflectorPlane,
      this._virtualCamera.viewMatrix,
      reflectorPlane
    );

    const clipPlane = new Cesium.Cartesian4(
      reflectorPlane.normal.x,
      reflectorPlane.normal.y,
      reflectorPlane.normal.z,
      reflectorPlane.distance
    );

    const projectionMatrix = Cesium.Matrix4.clone(
      this._virtualCamera.frustum.projectionMatrix
    );

    const q = new Cesium.Cartesian4(
      (Math.sign(clipPlane.x) + projectionMatrix[8]) / projectionMatrix[0],
      (Math.sign(clipPlane.y) + projectionMatrix[9]) / projectionMatrix[5],
      -1,
      (1.0 + projectionMatrix[10]) / projectionMatrix[14]
    );

    Cesium.Cartesian4.multiplyByScalar(
      clipPlane,
      2.0 / Cesium.Cartesian4.dot(clipPlane, q),
      clipPlane
    );

    projectionMatrix[2] = clipPlane.x;
    projectionMatrix[6] = clipPlane.y;
    projectionMatrix[10] = clipPlane.z + 1.0 - clipBias;
    projectionMatrix[14] = clipPlane.w;

    this._virtualCamera.frustum.customProjectionMatrix =
      Cesium.Matrix4.clone(projectionMatrix);

    return true;
  }

  preRender(scene) {
    const currnetDefaultViewCamera = scene._defaultView.camera;
    const currentShadowMap = scene.shadowMap;
    const currentGlobe = scene.globe.show;
    const currentShowSkirts = scene.globe.showSkirts;

    if (!this._updateVirtualCamera(scene._defaultView.camera)) {
      this._primitive.show = false;
      return;
    }

    this._primitive.show = false;
    scene._defaultView.camera = this._virtualCamera;
    scene.shadowMap = undefined;
    scene.globe.show = false;
    scene.globe.showSkirts = false;

    const context = scene.context;
    const width = context.drawingBufferWidth;
    const height = context.drawingBufferHeight;
    const hdr = scene.highDynamicRange;
    this._createFramebuffer(context, width, height, hdr);

    render(scene, this._colorFramebuffer);

    const appearance = this._primitive.appearance;

    const texture = Cesium.Texture.fromFramebuffer({
      context,
      framebuffer: this._colorFramebuffer,
    });
    texture.type = "sampler2D";

    this._material.uniforms.texture = texture;
    this._material.uniforms.time = performance.now() / 1000.0;
    this._material.uniforms.fixedFrameToEastNorthUpTransform =
      Cesium.Matrix4.toArray(
        this._getFixedFrameToEastNorthUpTransformFromWorldMatrix()
      );
    appearance.uniforms.reflectMatrix = Cesium.Matrix4.toArray(
      this._reflectMatrix
    );
    appearance.uniforms.reflectorProjectionMatrix = Cesium.Matrix4.toArray(
      this._reflectorProjectionMatrix
    );
    appearance.uniforms.reflectorViewMatrix = Cesium.Matrix4.toArray(
      this._reflectorViewMatrix
    );

    this._primitive.show = true;
    scene._defaultView.camera = currnetDefaultViewCamera;
    scene.shadowMap = currentShadowMap;
    scene.globe.show = currentGlobe;
    scene.globe.showSkirts = currentShowSkirts;
  }

  _createPrimitive(positions, extrudedHeight, flowDegrees) {
    const material = this._createReflectionWaterMaterial();
    this._material = material;

    const appearance = new Cesium.MaterialAppearance({
      material,
      vertexShaderSource: WaterAppearanceVS,
      translucent: true,
    });
    appearance.uniforms = {};
    appearance.uniforms.reflectMatrix = Cesium.Matrix4.toArray(
      this._reflectMatrix
    );
    appearance.uniforms.reflectorProjectionMatrix = Cesium.Matrix4.toArray(
      this._reflectorProjectionMatrix
    );
    appearance.uniforms.reflectorViewMatrix = Cesium.Matrix4.toArray(
      this._reflectorViewMatrix
    );

    const primitive = new Cesium.Primitive({
      geometryInstances: new Cesium.GeometryInstance({
        geometry: new Cesium.PolygonGeometry({
          polygonHierarchy: new Cesium.PolygonHierarchy(positions),
          perPositionHeight: true,
          extrudedHeight,
          stRotation: Cesium.Math.toRadians(flowDegrees),
          closeTop: true,
          closeBottom: false,
          vertexFormat: Cesium.VertexFormat.POSITION_NORMAL_AND_ST,
        }),
      }),
      appearance,
      asynchronous: false,
    });
    return primitive;
  }

  _getFixedFrameToEastNorthUpTransformFromWorldMatrix() {
    const eastNorthUpToFixedFrameTransform =
      Cesium.Transforms.eastNorthUpToFixedFrame(this._reflectorWorldPosition);

    const fixedFrameToEastNorthUpTransform = Cesium.Matrix4.inverse(
      eastNorthUpToFixedFrameTransform,
      new Cesium.Matrix4()
    );
    return fixedFrameToEastNorthUpTransform;
  }

  _createFramebuffer(context, width, height, hdr) {
    const colorTexture = this._colorTexture;
    if (
      Cesium.defined(colorTexture) &&
      colorTexture.width === width &&
      colorTexture.height === height &&
      this._hdr === hdr
    ) {
      return;
    }

    this._destroyResource();

    this._hdr = hdr;

    const pixelDatatype = hdr
      ? context.halfFloatingPointTexture
        ? Cesium.PixelDatatype.HALF_FLOAT
        : Cesium.PixelDatatype.FLOAT
      : Cesium.PixelDatatype.UNSIGNED_BYTE;
    this._colorTexture = new Cesium.Texture({
      context,
      width,
      height,
      pixelFormat: Cesium.PixelFormat.RGBA,
      pixelDatatype,
      sampler: new Cesium.Sampler({
        wrapS: Cesium.TextureWrap.CLAMP_TO_EDGE,
        wrapT: Cesium.TextureWrap.CLAMP_TO_EDGE,
        minificationFilter: Cesium.TextureMinificationFilter.LINEAR,
        magnificationFilter: Cesium.TextureMagnificationFilter.LINEAR,
      }),
    });

    this._depthStencilTexture = new Cesium.Texture({
      context,
      width,
      height,
      pixelFormat: Cesium.PixelFormat.DEPTH_STENCIL,
      pixelDatatype: Cesium.PixelDatatype.UNSIGNED_INT_24_8,
    });

    this._colorFramebuffer = new Cesium.Framebuffer({
      context,
      colorTextures: [this._colorTexture],
      depthStencilTexture: this._depthStencilTexture,
      destroyAttachments: false,
    });
  }

  _destroyResource() {
    this._colorTexture && this._colorTexture.destroy();
    this._depthStencilTexture && this._depthStencilTexture.destroy();
    this._colorFramebuffer && this._colorFramebuffer.destroy();

    this._colorTexture = undefined;
    this._depthStencilTexture = undefined;
    this._colorFramebuffer = undefined;
  }
}
