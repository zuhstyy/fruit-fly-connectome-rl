// Exercise the real adapter against a fake dependency, then inspect the bundled
// dependency's method metadata. This does not execute Celeste game physics.
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using Celeste.Mod.FlyBrainCeleste;

internal static class PracticeAdapterHarness
{
    private static int checks;
    private static void Check(bool condition, string reason)
    {
        checks++;
        if (!condition)
            throw new Exception(reason);
    }
    public static void Main(string[] args)
    {
        var practice = new PracticeStates();
        var inputs = new BotInputs();
        Check(!practice.Available, "disabled until initialization");
        practice.Initialize(typeof(PracticeAdapterHarness));
        Check(practice.Available, "required APIs found");
        Check(Celeste.Mod.SpeedrunTool.ModInterop.SaveLoadInterop.SaveLoadExports.Ignored ==
                  typeof(PracticeAdapterHarness),
              "HUD excluded from cloning");
        inputs.Restore(5);
        Check(practice.Capture(0, inputs), "capture entry");
        inputs.Restore(10);
        Check(practice.Capture(1, inputs), "capture frontier");
        Check(practice.Restore(0, inputs) && inputs.Capture() == 5, "entry button state");
        Check(practice.Restore(1, inputs) && inputs.Capture() == 10, "independent frontier");
        practice.ClearSlot(0);
        Check(!practice.Restore(0, inputs), "cleared entry unavailable");
        Celeste.Mod.SpeedrunTool.SaveLoad.SaveSlotsManager.Allow = false;
        Check(!practice.Capture(2, inputs), "rejected save is not remembered");
        Check(!practice.Restore(2, inputs), "rejected save cannot restore");
        practice.Clear();
        Check(!practice.Restore(1, inputs), "all saved button states released");

        using var zip = ZipFile.OpenRead(args[0]);
        var entry = zip.Entries.Single(e => e.Name == "SpeedrunTool.dll");
        using var bytes = new MemoryStream();
        using (var stream = entry.Open()) stream.CopyTo(bytes);
        bytes.Position = 0;
        using var pe = new PEReader(bytes);
        var metadata = pe.GetMetadataReader();
        var slots = metadata.TypeDefinitions.Select(metadata.GetTypeDefinition)
                        .Single(t => metadata.GetString(t.Namespace) ==
                                         "Celeste.Mod.SpeedrunTool.SaveLoad" &&
                                     metadata.GetString(t.Name) == "SaveSlotsManager");
        foreach (string name in new[] { "SaveStateTas", "LoadStateTas", "ClearStateTas" })
        {
            var method = slots.GetMethods()
                             .Select(metadata.GetMethodDefinition)
                             .Single(m => metadata.GetString(m.Name) == name &&
                                          (m.Attributes & MethodAttributes.Public) != 0);
            var signature = metadata.GetBlobReader(method.Signature);
            signature.ReadSignatureHeader();
            Check(signature.ReadCompressedInteger() == 1, name + " has one parameter");
            Check(signature.ReadSignatureTypeCode()
                      is SignatureTypeCode.Boolean or SignatureTypeCode.Void,
                  name + " return type");
            Check(signature.ReadSignatureTypeCode() == SignatureTypeCode.String,
                  name + " accepts a named slot");
        }
        var exports = metadata.TypeDefinitions.Select(metadata.GetTypeDefinition)
                          .Single(t => metadata.GetString(t.Name) == "SaveLoadExports");
        var ignore = exports.GetMethods()
                         .Select(metadata.GetMethodDefinition)
                         .Single(m => metadata.GetString(m.Name) == "IgnoreSaveState" &&
                                      AcceptsSystemType(metadata, m));
        var ignoreSignature = metadata.GetBlobReader(ignore.Signature);
        ignoreSignature.ReadSignatureHeader();
        Check(ignoreSignature.ReadCompressedInteger() == 2, "ignore API accepts type and flag");
        Check(ignoreSignature.ReadSignatureTypeCode() == SignatureTypeCode.Void,
              "ignore API return type");
        Console.WriteLine(
            $"{checks} C# practice adapter/API checks passed; no native game was run.");
    }
    private static bool AcceptsSystemType(MetadataReader reader, MethodDefinition method)
    {
        var signature = reader.GetBlobReader(method.Signature);
        signature.ReadSignatureHeader();
        if (signature.ReadCompressedInteger() != 2 ||
            signature.ReadSignatureTypeCode() != SignatureTypeCode.Void ||
            signature.ReadSignatureTypeCode() != SignatureTypeCode.TypeHandle)
            return false;
        var type = signature.ReadTypeHandle();
        if (type.Kind != HandleKind.TypeReference)
            return false;
        var reference = reader.GetTypeReference((TypeReferenceHandle)type);
        return reader.GetString(reference.Name) == "Type" &&
               reader.GetString(reference.Namespace) == "System" &&
               signature.ReadSignatureTypeCode() == SignatureTypeCode.Boolean;
    }
}

namespace Celeste.Mod.FlyBrainCeleste
{
// Only input storage is faked here. ButtonHistory is the production class.
internal sealed class BotInputs
{
    private readonly ButtonHistory history = new();
    public int Capture() => history.Capture();
    public void Restore(int bits) => history.Restore(bits);
}
}
namespace Celeste.Mod.SpeedrunTool.SaveLoad
{
internal static class SaveSlotsManager
{
    public static bool Allow = true;
    private static readonly HashSet<string> saved = new();
    public static bool SaveStateTas(string name) => Allow && saved.Add(name);
    public static bool LoadStateTas(string name) => saved.Contains(name);
    public static void ClearStateTas(string name) => saved.Remove(name);
}
}
namespace Celeste.Mod.SpeedrunTool.ModInterop
{
public static class SaveLoadInterop
{
    public static class SaveLoadExports
    {
        public static Type? Ignored;
        public static void IgnoreSaveState(Type type, bool unused = false) => Ignored = type;
    }
}
}
